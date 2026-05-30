import os, datetime, time
import twstock
import pandas as pd
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import pandas_market_calendars as mcal
import logging
# import yfinance as yf

# === 參數區 ===
CACHE_DIR = r'C:\Users\Jerry-yc.Liu\Desktop\script\cache'
YEARS = 2
MAX_WORKERS = 1
SLEEP_FETCH = 0.2
SLEEP_SAVE = 0.5

os.makedirs(CACHE_DIR, exist_ok=True)

# === 日誌與輸出設定 ===
log_path = os.path.join(CACHE_DIR, f"cache_update_{datetime.datetime.now(): %Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8")
    ]
)
logging.info(f"所有 print 內容將同步寫入 {log_path}")

# === 產業分類 ===
# industry_dict = {
#     '銀行業':['金融保險業'],
#     '醫療業':['生技醫療業'],
#     '科技業':['電子工業','半導體業','電腦及週邊設備業','光電業','通信網路業','電子零組件業','電子通路業','資訊服務業','其他電子業','電子商務','數位雲端'],     
#     '傳統產業':['航運業','電機機械','水泥工業','電器電纜','鋼鐵工業','汽車工業','食品工業','塑膠工業','紡織纖維','化學工業',
#                '造紙工業','橡膠工業','玻璃陶瓷','建材營造','貿易百貨','油電燃氣業','觀光餐旅','其他','綠能環保'],
#     '其他行業':['文化創意業','農業科技業','綜合','運動休閒','居家生活']  
# }

industry_dict = {
    '科技業':['電子工業','半導體業','電腦及週邊設備業','光電業','通信網路業','電子零組件業','電子通路業','資訊服務業','其他電子業','電子商務','數位雲端'],     
    '傳統產業':['航運業','電機機械','水泥工業','電器電纜','鋼鐵工業','汽車工業','食品工業','塑膠工業','紡織纖維','化學工業',
               '造紙工業','橡膠工業','玻璃陶瓷','建材營造','貿易百貨','油電燃氣業','觀光餐旅','綠能環保'],
}

from itertools import chain
target_group = set(chain.from_iterable(industry_dict.values()))

# === 讀取 no_data.txt 處理 ===
no_data_path = os.path.join(CACHE_DIR, "no_data.txt")
if os.path.exists(no_data_path):
    with open(no_data_path, "r", encoding="utf-8") as f:
        no_data_set = set(line.strip() for line in f if line.strip())
else:
    no_data_set = set()

# =============== 工具函式 ===============
def get_cache_file(code):
    return os.path.join(CACHE_DIR, f"{code}_{YEARS}y.pkl")

# === 判斷是否仍上市 ===
def is_listed(code):
    # remove_date 只要不是 None 或是空字串就代表以下市
    return not getattr(twstock.codes[code], 'removed_date', None)

# === 檢查快取是否已有資料 ===
def has_data(code):
    cache_file = get_cache_file(code)
    try:
        return os.path.exists(cache_file) and not pd.read_pickle(cache_file).empty
    except Exception as e:
        logging.error(f"has_data({code}) 讀取失敗: {e}")
        return False
    
def get_cache_latest_date(code):
    cache_file = get_cache_file(code)
    if not os.path.exists(cache_file):
        return None
    try:
        df = pd.read_pickle(cache_file)
        if df.empty:
            return None
        return df.index.max().date()
    except Exception as e:
        logging.error(f"get_cache_latest_date({code}) 讀取失敗: {e}")
        return None
    
def get_twstock_latest_date(code):
    stock = stock = twstock.Stock(code)
    if not stock.date:
        logging.warning(f"twstock 無資料: {code}")
        return None
    return stock.date[-1].date()

def get_last_trading_date():
    today = datetime.date.today()
    xtai = mcal.get_calendar('XTAI')
    schedule = xtai.schedule(start_date=today - datetime.timedelta(days=30), end_date=today)
    trading_days = schedule.index.date
    prev_days = [d for d in trading_days if d < today]
    if prev_days:
        return prev_days[-1]
    else:
        return None

def need_update(code, target_date):
    cache_file = get_cache_file(code)
    if not os.path.exists(cache_file):
        return True
    try:
        df = pd.read_pickle(cache_file)
        if df.empty:
            return True
        last_date = df.index.max().date()
        return last_date < target_date
    except:
        logging.error(f"need_update({code}) 讀取失敗: {e}")
        return True

# === 快取更新函式 ===
def update_cache(symbol, retry=2):
    cache_file = get_cache_file(symbol)
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=YEARS*365)
    old_df = pd.DataFrame()
    # 先讀舊資料
    if os.path.exists(cache_file):
        try:
            old_df = pd.read_pickle(cache_file)
            old_df = old_df[~old_df.index.duplicated(keep='last')]
        except Exception as e:
            logging.error(f"update_cache({symbol}) 舊資料讀取失敗: {e}")
    # 找出已經有的日期
    have_dates = set(old_df.index.date) if not old_df.empty else set()
    all_data = []
    stock = twstock.Stock(symbol)
    year, month = start_date.year, start_date.month
    while (year < today.year) or (year == today.year and month <= today.month):
        try:
            data = stock.fetch_from(year, month)
            if data:
                # 只保留還沒抓過的日期
                all_data.extend([d for d in data if d[0].date() not in have_dates])
        except Exception as e:
            print(f"{symbol} {YEARS}-{month:02d} 抓取失敗: {e}")
        month += 1
        if month > 12:
            month = 1
            year += 1
        time.sleep(0.2) # 避免被官方封鎖
    # 合併舊資料與新資料
    if all_data:
        df_new = pd.DataFrame(all_data, columns=['date','capacity','turnover','open','high','low','close','change','transaction'])
        df_new.set_index('date', inplace=True)
        df_new = df_new[~df_new.index.duplicated(keep='last')]
        df = pd.concat([old_df, df_new])
        df = df[~df.index.duplicated(keep='last')]
        df = df[df.index.date >= start_date]
        df.sort_index(inplace=True)
    else:
        df = old_df
    if df.empty:
        print(f"{symbol} 沒有資料，加入 no_data.txt")
        with open(no_data_path, "a", encoding="utf-8") as f:
            f.write(f"{symbol}\n")
        return

    try:
        df.to_pickle(cache_file)
        logging.info(f"{symbol} 快取已捕抓到 {df.index.max().strftime('%Y-%m-%d')}，共 {len(df)} 天")
    except Exception as e:
        logging.error(f"{symbol} 快取儲存失敗: {e}")
    
    time.sleep(SLEEP_SAVE)

def is_latest_data_available(stock_id, target_date):
    stock = twstock.Stock(stock_id)
    return stock.date[-1].date() == target_date

# =============== 主流程 ===============
def main():
    # === 股票過濾條件 ===
    # 只抓四大產業、上市櫃、未下市、非債券/ETF/受益憑證、且代號純數字的股票，且不再 no_data.txt
    filtered = [
        code for code, info in twstock.codes.items()
        if re.fullmatch(r'\d+', code) # 只保留純數字代號
        and hasattr(info, 'group') and hasattr(info, 'market')
        and info.market in ['上市','上櫃']
        and info.group in target_group
        and not code.endswith('U') # 排除受益憑證
        and not code.endswith('B') # 排除債券型 ETF
        and not code.startswith('7') # 排除債券、可轉債
        and not code.startswith('9') # 排除公司債
        and is_listed(code) # 至保留未下市
        and code not in no_data_set
    ]
    logging.info(f"\n三大產業過濾後股票數量: {len(filtered)}")

    # === 印出四大產業各自抓到幾支股票 ===
    industry_count = defaultdict(int)
    for code in filtered:
        group = twstock.codes[code].group
        for k, v in industry_dict.items():
            if group in v:
                industry_count[k] += 1
                break
    for k in industry_dict:
        logging.info(f"{k} 股票數量: {industry_count[k]}")

    # 1. 先檢查 twwstock 是否有今天的資料
    today = datetime.date.today()
    # 只挑選部分股票檢查 twstock 是否有今日資料，加速判斷
    sample_codes = filtered[:3] if len(filtered) >= 3 else filtered
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        sample_results = list(executor.map(get_twstock_latest_date, sample_codes))
    # 排除 None，僅針對有資料的股票判斷
    all_have_today = all((d is not None and d == today) for d in sample_results)

    if all_have_today:
        # 2. 檢查快取是否也有今天的資料
        cache_latest_dates = {code: get_cache_latest_date(code) for code in filtered}
        if all(d == today for d in cache_latest_dates.values()):
            logging.info(f"所有股票快取都已更新今日({today})，程式自動停止")
            return
        else:
            logging.info(f"twstock 已有今日({today})資料，但快取尚未更新，快取更新中...")
            target_date = today
    else:
        # 3. 沒有今天的資料，找上一個交易日
        last_trading_date = get_last_trading_date()
        logging.info(f"twstock 尚未有今日({today})資料，將以上個交易日({last_trading_date})為基準")
        cache_latest_dates = {code: get_cache_latest_date(code) for code in filtered}
        if all(d == last_trading_date for d in cache_latest_dates.values()):
            logging.info(f"所有股票皆已更新道上個交易日({last_trading_date})，程式自動停止。")
            return
        else:
            logging.info(f"快取尚未更新到上個交易日({last_trading_date})，快取更新中...")
            target_date = last_trading_date
    
    # 只抓需要更新的股票 (以 target_date 為基準)
    to_update = [code for code in filtered if need_update(code, target_date)]
    logging.info(f"\n需要抓股票的數量: {len(to_update)}")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(update_cache, to_update))

    logging.info("四大產業快取已捕抓到最新版")

    # === 再檢查哪些股票有資料(這裡只檢查快取，不會再更新) ===
    filtered_with_data = [code for code in filtered if has_data(code)]
    logging.info(f"\n四大產業有資料且未下市股票數量: {len(filtered_with_data)}")

    # 合併所有快取成一個大 pkl
    all_data = {}
    for code in filtered_with_data:
        cache_file = get_cache_file(code)
        if os.path.exists(cache_file):
            try:
                df = pd.read_pickle(cache_file)
                if not df.empty:
                    all_data[code] = df
            except Exception as e:
                logging.error(f"{code} 讀取失敗: {e}")

    big_pkl_path = os.path.join(CACHE_DIR, f"all_stock_{YEARS}y.pkl")
    pd.to_pickle(all_data, big_pkl_path)
    logging.info(f"已合併存成 {big_pkl_path}")

    # === 檢查是否有 open=high=low=close 的異常資料 ===
    logging.info("\n檢查是否有 open=high=low=close 的資料（疑似無交易日）...")
    for stock_id, df in all_data.items():
        bad_days = df[
            (df['open'] == df['high']) &
            (df['high'] == df['low']) &
            (df['low'] == df['close'])
        ]
        if not bad_days.empty:
            # logging.warning(f"{stock_id} 有 {len(bad_days)} 天 open=high=low=close")
            logging.debug(f"\n{bad_days[['open', 'capacity', 'transaction']].tail()}")

    logging.info(f"\n全部執行完畢，總耗時: {time.time() - start_time:.1f} 秒")
 
if __name__ == "__main__":
    main()