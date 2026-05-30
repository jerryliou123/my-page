"""
快速選單 - 股票分析系統統一入口
提供常用功能的快速訪問
"""

import os
import sys
from pathlib import Path
import datetime

# 先檢查依賴套件
def check_dependencies():
    """檢查必要套件是否已安裝"""
    missing = []

    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'twstock': 'twstock',
        'ta': 'ta',
    }

    for import_name, package_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("=" * 80)
        print("[X] 系統缺少必要套件")
        print("=" * 80)
        print()
        print("缺少以下套件:")

        for pkg in missing:
            print(f"  - {pkg}")

        print()
        print("請執行以下任一方法安裝:")
        print("  1. 雙擊執行: install_dependencies.bat")
        print("  2. 命令列執行: pip install -r requirements.txt")
        print("  3. 檢查套件: python check_dependencies.py")
        print()
        print("=" * 80)

        input("按 Enter 退出...")
        sys.exit(1)


# 執行檢查
check_dependencies()

# 檢查通過後 import pandas
import pandas as pd

# =============================================================================
# 設定
# =============================================================================

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / 'cache'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'models'

# =============================================================================
# 工具函數
# =============================================================================

def clear_screen():
    """清空螢幕"""
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except:
        print("\n" * 50)  # 如果清屏失敗，用換行代替


def print_banner():
    """顯示標題"""
    print("=" * 80)
    print("台股量化交易系統 - 快速選單 v2.0")
    print("=" * 80)


def check_data_status():
    """檢查資料狀態"""
    pkl_file = CACHE_DIR / 'all_stock_2y.pkl'

    if not pkl_file.exists():
        return "[X] 無資料", None

    try:
        file_size = pkl_file.stat().st_size / (1024 * 1024)
        mod_time = datetime.datetime.fromtimestamp(pkl_file.stat().st_mtime)
        days_old = (datetime.datetime.now() - mod_time).days

        if days_old == 0:
            status = "[OK] 最新"
        elif days_old <= 3:
            status = f"[!] {days_old}天前"
        else:
            status = f"[X] {days_old}天前"

        return status, file_size

    except Exception:
        return "[X] 損壞", None


def check_model_status():
    """檢查模型狀態"""

    if not MODEL_DIR.exists():
        return "[X] 無模型", None

    model_files = sorted([
        f for f in MODEL_DIR.iterdir()
        if f.name.startswith('model_') and f.suffix == '.pkl'
    ])

    if not model_files:
        return "[X] 無模型", None

    latest_model = model_files[-1]

    mod_time = datetime.datetime.fromtimestamp(latest_model.stat().st_mtime)
    days_old = (datetime.datetime.now() - mod_time).days

    if days_old == 0:
        status = f"[OK] 今天訓練"
    elif days_old <= 7:
        status = f"[!] {days_old}天前"
    else:
        status = f"[X] {days_old}天前"

    return status, latest_model.name


def get_latest_backtest():
    """取得最新回測結果"""

    if not OUTPUT_DIR.exists():
        return None

    csv_files = sorted([
        f for f in OUTPUT_DIR.iterdir()
        if f.name.startswith('backtest_') and f.suffix == '.csv'
    ])

    if not csv_files:
        return None

    return csv_files[-1]


def get_latest_prediction():
    """取得最新預測結果"""

    if not OUTPUT_DIR.exists():
        return None

    csv_files = sorted([
        f for f in OUTPUT_DIR.iterdir()
        if f.name.startswith('tomorrow_signals_') and f.suffix == '.csv'
    ])

    if not csv_files:
        return None

    return csv_files[-1]


def run_script(script_name, auto_choice=None):
    """執行 Python 腳本"""

    print(f"\n正在執行: {script_name}")
    print("=" * 80 + "\n")

    try:
        if auto_choice:
            temp_input = BASE_DIR / 'temp_input.txt'

            with open(temp_input, 'w', encoding='utf-8') as f:
                f.write(f"{auto_choice}\n")

            if sys.platform == 'win32':
                cmd = f'cd /d "{BASE_DIR}" && type "{temp_input}" | python "{script_name}"'
                os.system(cmd)
            else:
                cmd = f'cd "{BASE_DIR}" && cat "{temp_input}" | python "{script_name}"'
                os.system(cmd)

            try:
                temp_input.unlink()
            except Exception:
                pass

        else:
            os.system(f'python "{BASE_DIR / script_name}"')

    except Exception as e:
        print(f"\n[X] 執行錯誤: {e}")

    print("\n" + "=" * 80)
    input("\n按 Enter 返回主選單...")


def view_backtest_summary():
    """檢視回測結果摘要"""

    latest = get_latest_backtest()

    if not latest:
        print("\n[X] 找不到回測結果")
        print("請先執行選項 2 或 3 進行訓練和回測")
        input("\n按 Enter 返回...")
        return

    print(f"\n 回測結果摘要")
    print("=" * 80)
    print(f"檔案: {latest.name}")

    try:
        df = pd.read_csv(latest)

        print(f"\n訊號總數: {len(df):,}")
        print(f"獲利訊號: {df['win'].sum():,}")
        print(f"整體勝率: {df['win'].mean():.2%}")

        print(f"\n 分數區間勝率:")
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        df['score_bin'] = pd.cut(df['score'], bins=bins)

        for interval, grp in df.groupby('score_bin', observed=False):
            if len(grp) > 0:
                print(
                    f"  {interval.left:.1f}-{interval.right:.1f}: "
                    f"{len(grp):6,}筆, 勝率 {grp['win'].mean():6.2%}"
                )

        print(f"\n 獲利統計:")
        winning_trades = df[df['win'] == 1]

        if len(winning_trades) > 0:
            print(f"  平均獲利: {winning_trades['profit_pct'].mean():.2%}")
            print(f"  最大獲利: {winning_trades['profit_pct'].max():.2%}")

        print(f"\n 完整資料: {latest}")

    except Exception as e:
        print(f"[X] 讀取失敗: {e}")

    input("\n按 Enter 返回...")


def view_prediction_summary():
    """檢視預測結果摘要"""

    latest = get_latest_prediction()

    if not latest:
        print("\n[X] 找不到預測結果")
        print("請先執行選項 4 進行預測")
        input("\n按 Enter 返回...")
        return

    print(f"\n 今日預測摘要")
    print("=" * 80)
    print(f"檔案: {latest.name}")

    try:
        df = pd.read_csv(latest)

        print(f"\n找到 {len(df)} 支高品質訊號")

        if len(df) > 0:
            print("\n TOP 10 推薦:")
            print(f"{'代碼':<8} {'名稱':<10} {'分數':>8} {'收盤':>8} {'RSI':>6} {'趨勢':<6}")
            print("-" * 80)

            for idx, row in df.head(10).iterrows():
                trend = '多頭' if row['close'] > row['60MA'] else '空頭'

                print(
                    f"{row['code']:<8} "
                    f"{row['name']:<10} "
                    f"{row['score']:>7.2%} "
                    f"{row['close']:>8.1f} "
                    f"{row['RSI']:>6.1f} "
                    f"{trend:<6}"
                )

        print(f"\n 統計:")
        print(f"  平均分數: {df['score'].mean():.2%}")
        print(f"  平均RSI: {df['RSI'].mean():.1f}")
        print(f"  多頭趨勢: {(df['close'] > df['60MA']).sum()} 支")
        print(f"  空頭趨勢: {(df['close'] <= df['60MA']).sum()} 支")

        print(f"\n 完整資料: {latest}")

    except Exception as e:
        print(f"[X] 讀取失敗: {e}")

    input("\n按 Enter 返回...")


def quick_update_and_predict():
    """快速更新資料並預測"""

    print("\n 快速更新 + 預測流程")
    print("=" * 80)
    print("這將執行:")
    print("  1. 更新股票資料 (排除過舊股票)")
    print("  2. 使用現有模型進行預測")
    print("\n預計時間: 5-30 分鐘")

    confirm = input("\n確認執行？(y/N): ").strip().lower()

    if confirm != 'y':
        print("已取消")
        input("\n按 Enter 返回...")
        return

    print("\n" + "=" * 80)
    print("步驟 1/2: 更新資料")
    print("=" * 80)

    run_script('integrated_stock_analysis.py', auto_choice='1')

    print("\n" + "=" * 80)
    print("步驟 2/2: 預測明天訊號")
    print("=" * 80)

    run_script('predict_tomorrow.py', auto_choice='2\n50')

    print("\n[OK] 快速更新完成！")
    input("\n按 Enter 返回...")


# =============================================================================
# 主選單
# =============================================================================

def get_update_recommendation():
    """取得更新建議"""

    pkl_file = CACHE_DIR / 'all_stock_2y.pkl'

    today = datetime.datetime.now()
    is_weekend = today.weekday() >= 5

    if is_weekend:
        return "[OK] 週末無需更新 (股市休市)", False

    if not pkl_file.exists():
        return "[X] 必須更新 (無資料)", True

    try:
        mod_time = datetime.datetime.fromtimestamp(pkl_file.stat().st_mtime)
        days_old = (datetime.datetime.now() - mod_time).days

        if days_old == 0:
            return "[OK] 今天已更新，無需再次更新", False
        elif days_old == 1:
            return "[!] 建議更新 (昨天的資料)", True
        elif days_old <= 3:
            return f"[!] 建議更新 ({days_old}天前)", True
        else:
            return f"[X] 必須更新 ({days_old}天前，資料過舊)", True

    except Exception:
        return "[X] 必須更新 (檔案異常)", True


def main_menu():
    """顯示主選單"""

    while True:
        clear_screen()
        print_banner()

        data_status, data_size = check_data_status()
        model_status, model_name = check_model_status()
        update_rec, need_update = get_update_recommendation()

        print(f"\n[系統狀態]")
        print(f"  資料:  {data_status}" + (f" ({data_size:.1f} MB)" if data_size else ""))
        print(f"  模型:  {model_status}" + (f" ({model_name})" if model_name else ""))
        print(f"  建議:  {update_rec}")

        print(f"\n[快速功能]")

        if need_update:
            print("  1. 快速更新 + 預測 (推薦執行) ★")
        else:
            print("  1. 快速更新 + 預測")

        print("  2. 完整訓練 + 回測 (每月執行)")
        print("  3. 預測明天進場訊號")
        print("  4. 查看最新回測結果")
        print("  5. 查看最新預測結果")

        print(f"\n[進階功能]")
        print("  6. 更新資料 (僅抓取)")
        print("  7. 完整系統選單 (所有功能)")
        print("  8. 診斷工具")

        print("\n  0. 退出")

        print("\n" + "=" * 80)
        sys.stdout.flush()  # 確保所有輸出都顯示

        # 使用更可靠的輸入方式
        try:
            # 直接使用 input()，但先清空輸出緩衝
            print("請選擇 (0-8): ", end='', flush=True)
            choice = sys.stdin.readline().strip()
            if not choice:  # 如果讀到空字串 (EOF)
                print("\n[錯誤] 無法讀取輸入，程式結束")
                break
        except (EOFError, KeyboardInterrupt):
            print("\n\n程式中斷")
            break

        if choice == '0':
            print("\n 再見！")
            break

        elif choice == '1':
            quick_update_and_predict()

        elif choice == '2':
            print("\n[!] 這將執行完整的訓練和回測流程")
            print("預計時間: 2.5-3 小時")

            confirm = input("\n確認執行？(y/N): ").strip().lower()

            if confirm == 'y':
                run_script('integrated_stock_analysis.py', auto_choice='3')

        elif choice == '3':
            run_script('predict_tomorrow.py')

        elif choice == '4':
            view_backtest_summary()

        elif choice == '5':
            view_prediction_summary()

        elif choice == '6':
            print("\n選擇更新模式:")
            print("  1. 排除過舊股票 (快速)")
            print("  2. 包含過舊股票 (完整)")

            update_mode = input("\n選擇 (1/2): ").strip()

            if update_mode == '1':
                run_script('integrated_stock_analysis.py', auto_choice='1')
            elif update_mode == '2':
                run_script('integrated_stock_analysis.py', auto_choice='5')
            else:
                print("無效選項")
                input("\n按 Enter 返回...")

        elif choice == '7':
            run_script('integrated_stock_analysis.py')

        elif choice == '8':
            print("\n 診斷工具:")
            print("  1. 檢查資料完整性")
            print("  2. 診斷 all_stock_2y.pkl")

            diag_choice = input("\n選擇 (1/2): ").strip()

            if diag_choice == '1':
                run_script('integrated_stock_analysis.py', auto_choice='4')
            elif diag_choice == '2':
                run_script('integrated_stock_analysis.py', auto_choice='7')
            else:
                print("無效選項")
                input("\n按 Enter 返回...")

        else:
            print("\n[X] 無效選項，請重新選擇")
            input("\n按 Enter 返回...")


# =============================================================================
# 主程式
# =============================================================================

if __name__ == "__main__":
    try:
        main_menu()

    except KeyboardInterrupt:
        print("\n\n 程式已中斷")

    except Exception as e:
        print(f"\n[X] 執行錯誤: {e}")

        import traceback
        traceback.print_exc()

        input("\n按 Enter 退出...")
