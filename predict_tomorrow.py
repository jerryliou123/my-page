"""
明日股票進場訊號預測腳本
使用訓練好的模型預測明天適合進場的股票
"""

import os
import pickle
import pandas as pd
import numpy as np
import twstock
import datetime

# =============================================================================
# 設定
# =============================================================================

CACHE_DIR = r'D:\share\MT6815\py_script\mipc_script_2023_7e_full_GEN99R_W23.35.5\cache'
MODEL_DIR = os.path.join(CACHE_DIR, '../models')
OUTPUT_DIR = os.path.join(CACHE_DIR, '../output')

# 從 integrated_stock_analysis.py 複製特徵計算函數
exec(open('integrated_stock_analysis.py', encoding='utf-8').read().split('# =============================================================================\n# 主程式')[0])

# =============================================================================
# 主要功能
# =============================================================================

def load_latest_model():
    """載入最新的模型和縮放器"""

    model_files = sorted([
        f for f in os.listdir(MODEL_DIR)
        if f.startswith('model_') and f.endswith('.pkl')
    ])

    if not model_files:
        raise FileNotFoundError("找不到模型檔案！請先執行選項2訓練模型。")

    latest_model = model_files[-1]
    latest_scaler = latest_model.replace('model_', 'scaler_')

    print(f"[載入] 載入模型: {latest_model}")

    with open(os.path.join(MODEL_DIR, latest_model), 'rb') as f:
        clf = pickle.load(f)

    with open(os.path.join(MODEL_DIR, latest_scaler), 'rb') as f:
        scaler = pickle.load(f)

    return clf, scaler


def predict_tomorrow_signals(score_threshold=0.7, top_n=50):
    """
    預測明天的進場訊號

    Args:
        score_threshold: 最低分數門檻 (預設0.7，勝率74%)
        top_n: 最多回傳幾支股票
    """

    print("=" * 80)
    print("明日股票進場訊號預測")
    print("=" * 80)

    # 載入模型
    clf, scaler = load_latest_model()

    # 載入資料
    pkl_path = os.path.join(CACHE_DIR, f'all_stock_{YEARS}y.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError("找不到資料檔案！請先執行選項1抓取資料。")

    print(f"\n[資料] 載入股票資料...")
    all_data = pd.read_pickle(pkl_path)

    print(f"[OK] 載入 {len(all_data)} 支股票")

    # 預測每支股票的最新訊號
    results = []

    print(f"\n[分析] 分析最新技術指標...")

    for code in all_data:
        try:
            df = all_data[code]

            if df is None or len(df) < 60:
                continue

            # 計算技術指標
            df = add_features(df)

            if len(df) == 0:
                continue

            # 取最新一筆資料
            latest_idx = len(df) - 1
            features = make_features(df, latest_idx)

            if features is None or any(pd.isna(f) or np.isinf(f) for f in features):
                continue

            # 預測
            score = clf.predict_proba(
                scaler.transform([features])
            )[0][1]

            if score >= score_threshold:
                stock_info = twstock.codes.get(code)

                name = stock_info.name if stock_info else "未知"
                market = stock_info.market if stock_info else "未知"

                latest_data = df.iloc[-1]

                results.append({
                    'code': code,
                    'name': name,
                    'market': market,
                    'score': score,
                    'close': latest_data['close'],
                    'volume': latest_data['capacity'],
                    'date': latest_data.name.date(),
                    '5MA': latest_data['5MA'],
                    '20MA': latest_data['20MA'],
                    '60MA': latest_data['60MA'],
                    'RSI': latest_data['RSI'],
                    'MACD': latest_data['MACD'],
                })

        except Exception:
            continue

    if not results:
        print(f"\n[!] 沒有找到分數 ≥ {score_threshold} 的訊號")
        print("建議：")
        print(f"  1. 降低門檻 (如 {score_threshold - 0.1:.1f})")
        print("  2. 先執行選項1更新資料")

        return None

    # 排序並輸出
    df_signals = pd.DataFrame(results).sort_values(
        'score',
        ascending=False
    ).head(top_n)

    print(f"\n{'=' * 80}")
    print(f"[目標] 找到 {len(df_signals)} 支高品質訊號 (分數 ≥ {score_threshold})")
    print(f"{'=' * 80}")

    print("\n根據回測數據:")

    if score_threshold >= 0.9:
        print("  預期勝率: 98% [HOT]")
    elif score_threshold >= 0.8:
        print("  預期勝率: 90% [*]")
    elif score_threshold >= 0.7:
        print("  預期勝率: 74% [OK]")
    else:
        print("  預期勝率: 59-74%")

    print(f"\n[清單] 明日進場候選清單 (TOP {len(df_signals)}):\n")

    # 格式化輸出
    for idx, row in df_signals.iterrows():
        print(
            f"{row['code']:6s} "
            f"{row['name']:8s} | "
            f"分數:{row['score']:6.2%} | "
            f"收盤:{row['close']:6.1f} | "
            f"RSI:{row['RSI']:5.1f} | "
            f"趨勢:{'多頭' if row['close'] > row['60MA'] else '空頭'}"
        )

    # 儲存結果
    today = datetime.datetime.now().strftime('%Y%m%d')

    output_file = os.path.join(
        OUTPUT_DIR,
        f'tomorrow_signals_{today}.csv'
    )

    df_signals.to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig'
    )

    print(f"\n[OK] 結果已存: {output_file}")

    return df_signals


# =============================================================================
# 主程式
# =============================================================================

if __name__ == '__main__':

    print("\n" + "=" * 80)
    print("明日股票進場訊號預測系統")
    print("=" * 80)

    print("\n請選擇策略:")
    print("1. 保守策略 (分數 ≥ 0.9, 預期勝率 98%)")
    print("2. 積極策略 (分數 ≥ 0.8, 預期勝率 90%)")
    print("3. 平衡策略 (分數 ≥ 0.7, 預期勝率 74%)")
    print("4. 自訂門檻")

    choice = input("\n選擇 (1-4, 預設3): ").strip() or '3'

    threshold_map = {
        '1': 0.9,
        '2': 0.8,
        '3': 0.7
    }

    if choice in threshold_map:
        threshold = threshold_map[choice]

    elif choice == '4':
        threshold = float(input("請輸入分數門檻 (0.0-1.0): "))

    else:
        print("無效選項，使用預設值 0.7")
        threshold = 0.7

    top_n = input(f"\n最多顯示幾支股票？(預設50): ").strip()
    top_n = int(top_n) if top_n else 50

    try:
        df_signals = predict_tomorrow_signals(
            score_threshold=threshold,
            top_n=top_n
        )

        if df_signals is not None:
            print("\n" + "=" * 80)
            print("[建議] 使用建議:")
            print("=" * 80)
            print("1. 結合基本面分析篩選 (如營收成長、獲利能力)")
            print("2. 分散投資，不要全部重壓單一股票")
            print("3. 設定停損點 (建議3-5%)")
            print("4. 定期檢視持股，達獲利目標即出場")
            print(f"5. 獲利目標: {threshold * 100}% (參數設定)")

    except Exception as e:
        print(f"\n[X] 執行錯誤: {e}")

        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("程式執行完畢")
    print("=" * 80)
