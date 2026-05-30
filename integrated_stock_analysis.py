"""
整合股票分析系統 v2.0 - 精簡優化版
包含：資料抓取、技術分析、機器學習訓練、回測
"""

import os
import sys
import time
import datetime
import warnings
import re
import pickle
import gc

import numpy as np
import pandas as pd
import twstock
import ta
from tqdm import tqdm
from multiprocessing import cpu_count, freeze_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from concurrent.futures import ThreadPoolExecutor
import pandas_market_calendars as mcal
import logging
from itertools import chain
from scipy.signal import argrelmax, argrelmin

warnings.filterwarnings("ignore")

# ============================================================
# 參數設定
CACHE_DIR = r'D:\Jerry\script\my-page\cache'
YEARS, MAX_WORKERS, SLEEP_FETCH, SLEEP_SAVE = 2, 16, 0.03, 0.05
STALE_THRESHOLD = 90
user_choice, lookahead_days, profit_target, stop_loss, vol_times = 'all', 15, 0.07, 0.03, 0.05
ML_TEST_SIZE, ML_RANDOM_STATE, ML_N_ESTIMATORS, ML_MAX_DEPTH = 0.2, 42, 120, 10
ML_MIN_SAMPLES_SPLIT, ML_MIN_SAMPLES_LEAF, ML_SAVE_MODEL = 10, 5, True
MA_PERIODS, RSI_PERIOD = [5, 20, 60], 14

for d in [CACHE_DIR, os.path.join(CACHE_DIR, '../output'), os.path.join(CACHE_DIR, '../models')]:
    os.makedirs(d, exist_ok=True)

OUTPUT_DIR, MODEL_DIR = os.path.join(CACHE_DIR, '../output'), os.path.join(CACHE_DIR, '../models')

log_path = os.path.join(CACHE_DIR, f'integrated_{datetime.datetime.now():%Y%m%d_%H%M%S}.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
)

industry_dict = {
    '資訊服務': ['電子零組件', '半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'],
    '銀行業': ['金融保險業'],
    '傳統產業': ['航運業', '觀光業', '水泥工業', '塑膠工業', '鋼鐵工業', '汽車工業', '食品工業', '紡織纖維業', '電器電纜業', '電機機械業', '機械工業', '建材營造業', '化學工業', '造紙工業', '橡膠工業', '營建業', '玻璃陶瓷業', '石油化學業', '其他'],
    'ETF': ['ETF', '指數股票型基金']
}

target_groups = set(chain.from_iterable(industry_dict.values()))


def load_file_set(path):
    return set(
        line.strip().split()[0] if ' ' in line else line.strip()
        for line in open(path, encoding='utf-8') if line.strip()
    ) if os.path.exists(path) else set()


no_data_set = load_file_set(os.path.join(CACHE_DIR, "no_data.txt"))
stale_stocks_set = load_file_set(os.path.join(CACHE_DIR, "stale_stocks.txt"))


def reload_filter_sets():
    """重新載入過濾清單"""
    global no_data_set, stale_stocks_set
    no_data_set = load_file_set(os.path.join(CACHE_DIR, "no_data.txt"))
    stale_stocks_set = load_file_set(os.path.join(CACHE_DIR, "stale_stocks.txt"))
    return no_data_set, stale_stocks_set


# ============================================================
# 工具函式
get_cache_file = lambda code: os.path.join(CACHE_DIR, f"{code}_{YEARS}y.pkl")
is_listed = lambda code: not getattr(twstock.codes[code], 'removed_date', None)


def has_data(code):
    try:
        return os.path.exists(get_cache_file(code)) and not pd.read_pickle(get_cache_file(code)).empty
    except Exception:
        return False


def get_cache_date(code):
    try:
        df = pd.read_pickle(get_cache_file(code))
        return df.index.max().date() if not df.empty else None
    except Exception:
        return None


def get_last_trading_date():
    today = datetime.date.today()
    schedule = mcal.get_calendar('XTAI').schedule(
        start_date=today - datetime.timedelta(days=30),
        end_date=today
    )
    prev_days = [d for d in schedule.index.date if d < today]
    return prev_days[-1] if prev_days else None


def check_optimal_time():
    now, today = datetime.datetime.now(), datetime.date.today()
    if today.weekday() >= 5:
        logging.warning(f"⚠️ 週末非交易日")
        return False
    hour = now.hour
    if 9 <= hour < 14:
        logging.warning(f"⏳ 盤中時間，建議17:00後執行")
        return False
    logging.info(f"📌 當前時間: {now.strftime('%H:%M')} - {'最佳' if 17 < hour < 24 else '可用'}擷取時間")
    return True


def update_cache(symbol):
    cache_file = get_cache_file(symbol)
    today = datetime.date.today()
    keep_start = today - datetime.timedelta(days=YEARS * 365)

    old_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            old_df = pd.read_pickle(cache_file)
            old_df = old_df[~old_df.index.duplicated(keep='last')]
        except Exception as e:
            logging.error(f"{symbol} 讀取失敗: {e}")

    fetch_start = old_df.index.max().date().replace(day=1) if not old_df.empty else keep_start
    have_dates = set(old_df.index.date) if not old_df.empty else set()
    all_data = []
    stock = twstock.Stock(symbol)

    year, month = fetch_start.year, fetch_start.month
    while (year < today.year) or (year == today.year and month <= today.month):
        try:
            data = stock.fetch_from(year, month)
            if data:
                all_data.extend([d for d in data if d[0].date() not in have_dates])
        except Exception as e:
            logging.error(f"{symbol} {year}-{month:02d} 失敗: {e}")
        month = month % 12 + 1
        year += month == 1
        time.sleep(SLEEP_FETCH)

    if all_data:
        df_new = pd.DataFrame(
            all_data,
            columns=['date', 'capacity', 'turnover', 'open', 'high', 'low', 'close', 'change', 'transaction']
        )
        df_new.set_index('date', inplace=True)
        combined = pd.concat([old_df, df_new])
        combined = combined[~combined.index.duplicated(keep='last')]
        df = combined[combined.index.date >= keep_start].sort_index()
    else:
        df = old_df[old_df.index.date >= keep_start] if not old_df.empty else pd.DataFrame()

    if df.empty:
        no_data_file = os.path.join(CACHE_DIR, "no_data.txt")
        existing = load_file_set(no_data_file)
        if symbol not in existing:
            with open(no_data_file, "a", encoding='utf-8') as f:
                f.write(f"{symbol}\n")
        return

    df.to_pickle(cache_file)
    days_old = (today - df.index.max().date()).days
    logging.info(f"{'⚠️' if days_old > 7 else '✅'} {symbol} 更新至 {df.index.max().date()}，折舊 {days_old}天")
    time.sleep(SLEEP_SAVE)


def check_completeness(filtered):
    logging.info("\n" + "="*80 + "\n 檢查資料完整性...\n" + "="*80)
    today = datetime.date.today()
    target_start = today - datetime.timedelta(days=YEARS * 365)
    stats = {
        'total': len(filtered),
        'complete': 0,
        'incomplete': 0,
        'outdated': 0,
        'no_cache': 0,
        'details': []
    }

    for code in tqdm(filtered, desc="檢查完整性"):
        cache_file = get_cache_file(code)
        name = twstock.codes.get(code).name if twstock.codes.get(code) else "未知"

        if not os.path.exists(cache_file):
            stats['no_cache'] += 1
            stats['details'].append({'code': code, 'name': name, 'status': 'NO_CACHE'})
            continue

        try:
            df = pd.read_pickle(cache_file)
            if df.empty:
                stats['no_cache'] += 1
                continue
            earliest, latest = df.index.min().date(), df.index.max().date()
            is_outdated = latest < today - datetime.timedelta(days=7)
            is_incomplete = earliest > target_start
            if is_outdated:
                status = 'OUTDATED'
            elif is_incomplete:
                status = 'INCOMPLETE'
            else:
                status = 'COMPLETE'
            stats[status.lower()] += 1
            stats['details'].append({
                'code': code,
                'name': name,
                'status': status,
                'date_range': f"{earliest} ~ {latest}",
                'days_from_today': (today - latest).days
            })
        except Exception as e:
            logging.error(f"{code} 檢查錯誤: {e}")

    logging.info(f"\n 統計: {stats['total']} | 完整: {stats['complete']} | ⚠️不完整: {stats['incomplete']} | ❌過時: {stats['outdated']}")
    pd.DataFrame(stats['details']).to_csv(
        os.path.join(CACHE_DIR, f"completeness_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    return stats


def get_filtered_codes(ignore_stale=False):
    """取得過濾後的股票代碼清單"""
    return [
        code for code, info in twstock.codes.items()
        if re.fullmatch(r'\d+', code)
        and hasattr(info, 'group')
        and hasattr(info, 'market')
        and info.market in ['上市', '上櫃']
        and info.group in target_groups
        and not code.endswith('00')
        and not code.endswith('B')
        and not code.startswith('7')
        and not code.startswith('9')
        and is_listed(code)
        and code not in no_data_set
        and (ignore_stale or code not in stale_stocks_set)
    ]


def capture_stock_data(ignore_stale=False):
    global stale_stocks_set, no_data_set
    start_time = time.time()
    logging.info("\n" + "="*60 + "\n 資料擷取開始\n" + "="*60)
    check_optimal_time()

    filtered = get_filtered_codes(ignore_stale=ignore_stale)

    if not ignore_stale and len(stale_stocks_set) > 0:
        logging.warning(f"⚠️ 已排除 {len(stale_stocks_set)} 支過舊股票，支援擷取股票 (資料超過 {STALE_THRESHOLD} 天)")
        logging.info("若要強制擷取請選擇『忽略過舊股票選項』")
    elif ignore_stale and len(stale_stocks_set) > 0:
        logging.info(f"✅ 正在更新過舊股票清單: {len(stale_stocks_set)} 支過舊股票")

    logging.info(f"\n目前股票數量: {len(filtered)}")
    check_completeness(filtered)

    today = datetime.date.today()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        cache_dates = dict(zip(filtered, executor.map(get_cache_date, filtered)))

    target_date = today if any(d == today for d in list(cache_dates.values())[:3]) else get_last_trading_date()
    if target_date is None:
        target_date = today

    to_update = [c for c in filtered if not cache_dates.get(c) or cache_dates[c] < target_date]

    if not to_update:
        logging.info("✅ 全部最新，無需更新")
        return

    logging.info(f"需要更新: {len(to_update)}/{len(filtered)}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(update_cache, to_update))

    all_data = {c: pd.read_pickle(get_cache_file(c)) for c in filtered if has_data(c)}
    pd.to_pickle(all_data, os.path.join(CACHE_DIR, f'all_stock_{YEARS}y.pkl'))

    if ignore_stale:
        logging.info("📌 重新檢視過舊股票清單...")
        today = datetime.date.today()
        new_stale = []
        for code in filtered:
            cache_date = get_cache_date(code)
            if cache_date and (today - cache_date).days > STALE_THRESHOLD:
                stock_name = twstock.codes.get(code).name if twstock.codes.get(code) else "未知"
                market = twstock.codes.get(code).market if twstock.codes.get(code) else "未知"
                new_stale.append(f"{code} {stock_name} {market} {cache_date} {(today - cache_date).days}天")
        stale_file = os.path.join(CACHE_DIR, "stale_stocks.txt")
        with open(stale_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(new_stale))
        reload_filter_sets()

    logging.info(f"✅ 完成! 花費 {time.time() - start_time:.1f}秒")


# ============================================================
# 技術指標與型態

def add_features(df):
    df = df.copy()
    for p, n in zip(MA_PERIODS, ['5MA', '20MA', '60MA']):
        df[n] = df['close'].rolling(p).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['Histogram'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 9, 3)
    df['K'] = stoch.stoch()
    df['D'] = stoch.stoch_signal()
    return df.dropna()


def detect_pattern(df, i, pattern_type='w_bottom'):
    if i < 20:
        return False
    closes = df['close'].iloc[i-40:i+1].values
    lows = df['low'].iloc[i-40:i+1].values
    highs = df['high'].iloc[i-40:i+1].values
    if pattern_type == 'w_bottom':
        min1 = np.argmin(lows[2:20])
        min2 = np.argmin(lows[:20]) + 20
        if abs(lows[min1] - lows[min2]) / max(lows[min1], 1e-6) < 0.03 and closes[-1] > np.max(closes[min1:min2+1]):
            return True
    elif pattern_type == 'triangle':
        h_idx = argrelmax(highs, np.greater, order=2)[0]
        l_idx = argrelmin(lows, np.less, order=2)[0]
        if len(h_idx) >= 2 and len(l_idx) >= 2:
            return highs[h_idx[-1]] < highs[h_idx[0]] and lows[l_idx[-1]] > lows[l_idx[0]]
        return False
    return False

pattern_funcs = {
    p: (lambda df, i, p=p: detect_pattern(df, i, p))
    for p in ['w_bottom', 'triangle', 'cup_handle', 'inverse_head']
}


def make_features(df, i):
    try:
        cur, prv = df.iloc[i], df.iloc[i-1]

        signals = [
            int(cur['close'] > cur['open'] and cur['high'] - cur['close'] > (cur['close'] - cur['open'])),
            int(cur['5MA'] > cur['20MA'] and cur['20MA'] > cur['60MA']),
            int(prv['RSI'] < 30 and cur['RSI'] > 30),
            int(prv['MACD'] < prv['Signal'] and cur['MACD'] > cur['Signal']),
            int(prv['K'] < prv['D'] and cur['K'] > cur['D'] and cur['K'] < 30),
            int(prv['capacity'] > 0 and cur['capacity'] >= vol_times * prv['capacity'])
        ]

        bullish = int(cur['5MA'] > cur['20MA'] > cur['60MA'])
        trends = [
            bullish,
            int(cur['5MA'] > prv['5MA']),
            int(cur['20MA'] > prv['20MA']),
            int(cur['60MA'] > prv['60MA']),
            int(cur['RSI'] > 50)
        ]

        ratios = [
            cur['close'] / (cur[n] + 1e-6)
            for n in ['5MA', '20MA', '60MA']
        ] + [
            cur['capacity'] / (df['capacity'].iloc[i-w:i].mean() + 1e-6)
            for w in [5, 20]
        ]

        patterns = [int(f(df, i)) for f in pattern_funcs.values()]
        return signals + [cur['RSI'], cur['MACD'], cur['K'], cur['D']] + ratios + trends + patterns
    except Exception:
        return None

feature_names = [
    'K_pattern', 'ma_cross', 'rsi_signal', 'macd_signal', 'kd_cross', 'vol_cross',
    'RSI', 'MACD', 'K', 'D', 'close/5MA', 'close/20MA', 'close/60MA', 'vol/5MA', 'vol/20MA',
    'bullish_ma', 'ma5 up', 'ma20 up', 'ma60 up', 'rsi 50'
] + list(pattern_funcs.keys())


def calc_win(entry, future_closes):
    if len(future_closes) == 0:
        return 0
    max_gain = np.max(future_closes) / entry - 1
    max_loss = np.min(future_closes) / entry - 1
    return int((max_loss > -stop_loss and max_gain >= profit_target) or max_gain >= profit_target * 0.5)


def make_dataset(df):
    X, y = [], []
    prices = df['close'].values
    for i in range(60, len(df) - lookahead_days):
        feat = make_features(df, i)
        if feat and not any(pd.isna(f) or np.isinf(f) for f in feat):
            X.append(feat)
            y.append(calc_win(prices[i], prices[i+1:i+1+lookahead_days]))
    return X, y


def process_stock(code, data):
    try:
        if code not in data:
            logging.warning(f"{code} 不在資料集中")
            return code, None

        df = data[code]
        if df is None or len(df) == 0:
            logging.warning(f"{code} 資料為空")
            return code, None

        if len(df) < 60:
            logging.warning(f"{code} 資料不足 ({len(df)} 筆 < 60)")
            return code, None

        df_with_features = add_features(df)
        if df_with_features is None or len(df_with_features) == 0:
            logging.warning(f"{code} 計算指標結果無資料")
            return code, None

        return code, df_with_features
    except Exception as e:
        logging.error(f"{code} 處理失敗: {type(e).__name__}: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return code, None


def analyze_stocks():
    logging.info("\n" + "="*80 + "\n開始技術分析、ML訓練 + 回測\n" + "="*80)

    pkl_path = os.path.join(CACHE_DIR, f'all_stock_{YEARS}y.pkl')
    if not os.path.exists(pkl_path):
        logging.error("找不到快取! 請先執行資料抓取")
        return

    all_data = pd.read_pickle(pkl_path)
    logging.info(f"從 pkl 讀取資料量 {len(all_data)} 支股票")

    target = set(chain.from_iterable(industry_dict.values())) if user_choice == 'all' else set(industry_dict.get(user_choice, []))
    logging.info(f"目標產業分類: {len(target)} 個")

    tw_list = []
    no_group = 0
    not_in_target = 0
    for c in all_data:
        stock_info = twstock.codes.get(c)
        if stock_info is None or not hasattr(stock_info, 'group'):
            no_group += 1
            logging.debug(f"{c} 無產業分類資訊")
            continue
        code_group = stock_info.group
        if code_group not in target:
            not_in_target += 1
            logging.debug(f"{c} 產業 '{code_group}' 不在目標中")
            continue
        tw_list.append(c)

    logging.info(f"產業分類過濾: {len(all_data)} 支 -> {len(tw_list)} 支 (排除: #group={no_group}, 不在目標={not_in_target})")

    if len(tw_list) == 0:
        logging.error("❌ 所有股票都被產業分類過濾掉了!")
        logging.error(f"📌 目標產業: {list(target)[:10]}...")
        actual_groups = set()
        for c in list(all_data.keys())[:20]:
            if hasattr(twstock.codes.get(c), 'group'):
                actual_groups.add(twstock.codes[c].group)
        logging.error(f"❌ 實際產業分類: {list(actual_groups)}")
        return

    logging.info(f"使用 ThreadPoolExecutor 處理 {len(tw_list)} 支股票...")
    results = []
    with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as executor:
        futures = [executor.submit(process_stock, code, all_data) for code in tw_list]
        for future in tqdm(futures, total=len(tw_list), desc="評估特徵"):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"處理股票失敗: {e}")

    all_df = {c: d for c, d in results if d is not None}

    total_processed = len(results)
    successful = len(all_df)
    failed = total_processed - successful
    logging.info(f"小計結果: 共處理 {total_processed} 支 | 成功 {successful} | 失敗 {failed} 支")

    if not all_df:
        logging.error("❌ 無可用股票資料!")
        logging.error("可能原因:")
        logging.error(" 1. all_stock_2y.pkl 檔案損毀或為空")
        logging.error(" 2. 所有股票資料筆數 < 60")
        logging.error(" 3. 計算技術指標後全都被 dropna 移除")
        logging.error(" 4. 先前抓取資料 1 5 年資料缺少")
        logging.error("   請於 cache/all_stock_2y.pkl 另行檢查何者有效")
        return

    del results, all_data
    gc.collect()

    logging.info("準備訓練資料...")
    def make_dataset_wrapper(code):
        # 診斷用: 記錄每個股票的資料長度與產生的樣本數
        if code not in all_df:
            return [], []
        df = all_df[code]
        try:
            logging.info(f"製作訓練資料: {code} 資料長度={len(df)}")
            X, y = make_dataset(df)
            logging.info(f"製作訓練資料完成: {code} -> 樣本數={len(X)} 正樣本={sum(y) if y else 0}")
            return X, y
        except Exception as e:
            logging.error(f"製作訓練資料失敗: {code} {e}")
            return [], []

    train_data = []
    with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as executor:
        futures = [executor.submit(make_dataset_wrapper, code) for code in all_df.keys()]
        for future in tqdm(futures, total=len(all_df), desc="製作訓練資料"):
            try:
                train_data.append(future.result())
            except Exception as e:
                logging.error(f"製作訓練資料失敗: {e}")
                train_data.append(([], []))

    X_all, y_all = [], []
    for X, y in train_data:
        X_all.extend(X)
        y_all.extend(y)

    if not X_all:
        logging.error("無訓練資料")
        return

    logging.info(f"樣本數量: {len(X_all)} | 正樣本: {sum(y_all)} ({sum(y_all)/len(y_all)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        stratify=y_all
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=ML_N_ESTIMATORS,
        max_depth=ML_MAX_DEPTH,
        min_samples_split=ML_MIN_SAMPLES_SPLIT,
        min_samples_leaf=ML_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=ML_RANDOM_STATE,
        class_weight='balanced'
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    logging.info("\n" + "="*80 + "\n訓練結果\n" + "="*80)
    logging.info(f"準確率: {clf.score(X_test_sc, y_test):.4f}")
    try:
        logging.info(f"AUC: {roc_auc_score(y_test, clf.predict_proba(X_test_sc)[:, 1]):.4f}")
    except Exception:
        pass

    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"\n混淆矩陣:\n TN:{cm[0,0]:6d} | FP:{cm[0,1]:6d}\n FN:{cm[1,0]:6d} | TP:{cm[1,1]:6d}")
    logging.info("\n" + classification_report(y_test, y_pred, target_names=['負', '正']))

    importance = pd.DataFrame(
        {'feature': feature_names, 'importance': clf.feature_importances_}
    ).sort_values('importance', ascending=False)
    logging.info("特徵重要性 TOP 10:")
    for _, row in importance.head(10).iterrows():
        logging.info(f" {row['feature']:25s}: {row['importance']:.6f}")

    if ML_SAVE_MODEL:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(MODEL_DIR, f"model_{ts}.pkl"), 'wb') as f:
            pickle.dump(clf, f)
        with open(os.path.join(MODEL_DIR, f"scaler_{ts}.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"模型已存: model_{ts}.pkl")

    def backtest_stock(code):
        try:
            df = all_df[code]
            results = []
            prices = df['close'].values
            dates = df.index
            for i in range(60, len(df) - lookahead_days):
                feat = make_features(df, i)
                if feat and not any(pd.isna(f) or np.isinf(f) for f in feat):
                    score = clf.predict_proba(scaler.transform([feat]))[0][1]
                    future = df.loc[(dates > dates[i]) & (dates <= dates[i] + pd.Timedelta(days=lookahead_days)), 'close']
                    if len(future) > 0:
                        profit = (future - prices[i]) / prices[i] * 100
                        results.append((code, dates[i], score, calc_win(prices[i], future.values), profit.max(), profit.idxmax()))
            return results
        except Exception:
            return []

    logging.info("回測中...")
    backtest_res = []
    with ThreadPoolExecutor(max_workers=min(cpu_count(), 4)) as executor:
        futures = [executor.submit(backtest_stock, code) for code in all_df.keys()]
        for future in tqdm(futures, total=len(all_df), desc="回測"):
            try:
                backtest_res.append(future.result())
            except Exception as e:
                logging.error(f"回測錯誤: {e}")
                backtest_res.append([])

    all_results = [r for res in backtest_res for r in res]
    if not all_results:
        logging.warning("無回測結果")
        return

    df_bt = pd.DataFrame(all_results, columns=['code', 'entry_date', 'score', 'win', 'profit_pct', 'max_profit_date'])
    logging.info("\n" + "="*80 + "\n回測結果\n" + "="*80)
    logging.info(f"總樣本: {len(df_bt)} | 勝率: {df_bt['win'].sum()} / {len(df_bt)} | 勝率: {df_bt['win'].mean():.2%}")

    score_bins = pd.cut(df_bt['score'], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for interval, grp in df_bt.groupby(score_bins, observed=False):
        if len(grp) > 0:
            logging.info(f"分數 {interval.left:.1f}-{interval.right:.1f}: {len(grp)} 支, 勝率 {grp['win'].mean():.2%}")

    output = os.path.join(OUTPUT_DIR, f"backtest_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv")
    df_bt.to_csv(output, index=False, encoding='utf-8-sig')
    logging.info(f"結果已存: {output}")


if __name__ == '__main__':
    freeze_support()

    if sys.platform == 'win32':
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass

    print("="*80 + "\n整合股票分析系統 v2.0 - 精簡版\n" + "="*80)
    print(f"\n目前狀態")
    print(f" - no_data.txt: {len(no_data_set)} 支 (無資料)")
    print(f" - stale_stocks.txt: {len(stale_stocks_set)} 支 (資料過舊)")
    print(f" - 資料目標股票數: {len(get_filtered_codes())} 支")
    print(f" - 若忽略過舊: {len(get_filtered_codes(ignore_stale=True))} 支")
    print("\n[選單]")
    print("1. 資料擷取 (排除過舊股票)")
    print("2. 技術分析 + ML訓練 + 回測")
    print("3. 完整執行 (排除過舊股票)")
    print("4. 檢查完整性")
    print("5. 資料擷取 (包含過舊股票) [推薦]")
    print("6. 清空過濾清單 (重置 no_data.txt 和 stale_stocks.txt)")
    print("7. 診斷 all_stock_2y.pkl 檔案")

    mode = input("\n選項 (1-7 [預設3]): ").strip() or '3'

    if mode == '1':
        capture_stock_data(ignore_stale=False)
    elif mode == '2':
        analyze_stocks()
    elif mode == '3':
        capture_stock_data(ignore_stale=False)
        analyze_stocks()
    elif mode == '4':
        check_completeness(get_filtered_codes(ignore_stale=True))
    elif mode == '5':
        print(f"\n⚠️ 即將更新所有股票 (包含過舊的 {len(stale_stocks_set)} 支股票)")
        confirm = input("確認執行? (y/N): ").strip().lower()
        if confirm == 'y':
            capture_stock_data(ignore_stale=True)
            reload_filter_sets()
            print(f"\n✅ 過濾清單已更新")
            print(f" - no_data.txt: {len(no_data_set)} 支")
            print(f" - stale_stocks.txt: {len(stale_stocks_set)} 支")
        else:
            print("已取消")
    elif mode == '6':
        print("\n⚠️ 即將清空過濾清單，這會讓程式重新嘗試所有股票")
        confirm = input("確認執行? (y/N): ").strip().lower()
        if confirm == 'y':
            import shutil
            for f in ['no_data.txt', 'stale_stocks.txt']:
                path = os.path.join(CACHE_DIR, f)
                if os.path.exists(path):
                    backup = path.replace('.txt', f'_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt')
                    shutil.copy(path, backup)
                    open(path, 'w', encoding='utf-8').close()
                    print(f"✅ {f} 已清空 (備份至 {os.path.basename(backup)})")
            print("\n清空完成，請重新執行程式")
        else:
            print("已取消")
    elif mode == '7':
        print("\n診斷 all_stock_2y.pkl 檔案...")
        pkl_path = os.path.join(CACHE_DIR, f'all_stock_{YEARS}y.pkl')
        if not os.path.exists(pkl_path):
            print(f"❌ 檔案不存在: {pkl_path}")
            print("請先執行選項 1 或 5 來生成檔案")
        else:
            try:
                file_size = os.path.getsize(pkl_path) / (1024 * 1024)
                print(f"✅ 檔案存在: {pkl_path}")
                print(f"📦 檔案大小: {file_size:.2f} MB")
                all_data = pd.read_pickle(pkl_path)
                print(f"📊 包含 {len(all_data)} 支股票")
                data_counts = {code: len(df) for code, df in all_data.items() if df is not None and len(df) > 0}
                if data_counts:
                    print(f"\n資料筆數統計:")
                    print(f"  最少: {min(data_counts.values())} 筆")
                    print(f"  最多: {max(data_counts.values())} 筆")
                    print(f"  平均: {sum(data_counts.values())/len(data_counts):.1f} 筆")
                    print(f"  不足筆數: {sum(1 for v in data_counts.values() if v < 60)} 支")
                print(f"\n隨機顯示 5 支股票的範例")
                for code, df in list(all_data.items())[:5]:
                    if df is not None and len(df) > 0:
                        name = twstock.codes.get(code).name if twstock.codes.get(code) else "未知"
                        print(f" {code} {name}: {len(df)} 筆, {df.index.min().date()} ~ {df.index.max().date()}")
                    else:
                        print(f" {code}: 無資料")
            except Exception as e:
                print(f"❌ 讀取失敗: {e}")
    else:
        print("無效選項")

    print("\n" + "="*80 + "\n程式執行完畢\n" + "="*80)
