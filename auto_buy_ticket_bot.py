from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
from selenium.common.exceptions import TimeoutException
import os, time, random

chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

max_tickets=2 # 要搶的張數
save_path = r"C:\Users\LYC\Desktop\Tool\workspace\debug_checkbox.html"

# chrome_options.add_argument("--headless")  # 例如設定為無頭模式
try: # 主要流程
    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)
    actions = ActionChains(driver)

    driver.get('https://kktix.com/')

    # 登入
    login_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "登入")))
    login_button.click()

    # 自動登入
    account = wait.until(EC.element_to_be_clickable((By.XPATH,'//*[@id="user_login"]')))
    account.clear()
    account.send_keys('jerryliou123@gmail.com')
    password = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="user_password"]')))
    password.clear()
    password.send_keys('Hnaka123')
    login_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="new_user"]/input[3]')))
    login_btn.click()

    time.sleep(random.uniform(2, 3))  # 隨機等待 2~3 秒，模擬真人行為

    # 點選活動
    activity_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="react-main-container"]/div/div/div[2]/div[2]/section[3]/ul/li[1]/a/figure/figcaption/div/div')))
    actions.move_to_element(activity_btn).pause(1).click().perform() # 取代 activity_btn.click()

    # 下一步（進入選票頁）
    next_btn_initial = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "下一步"))) # 用文字比較不會變
    actions.move_to_element(next_btn_initial).pause(1).click().perform() # 取代 next_btn.click()

    # === 選擇票種 ===
    print("⌛ 嘗試抓取票種區塊...")
    ticket_blocks = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.ticket-unit")))
    print(f"✅ 找到 {len(ticket_blocks)} 個票種區塊")

    selected = 0 # 計數選到幾張票
    for block in ticket_blocks:
        # 抓 + 按鈕 (如果沒有就是空 list)
        plus_buttons = block.find_elements(By.XPATH, ".//button[contains(@class,'plus')]")
        if not plus_buttons:
            print("⚠️ 此票種沒有 plus 按鈕，可能已售完")
            continue
        # 如果找到 + 按鈕，嘗試點擊
        for _ in range(max_tickets - selected):  # 只補到還缺的數量
            plus_buttons[0].click()
            selected += 1
            print(f"🎫 已成功選 {selected} 張票")
            # 隨機停頓 1~3 秒，模擬人類操作
            time.sleep(random.uniform(1, 2))
            if selected >= max_tickets:
                break
            print("✅ 已達到目標票數，停止選票")   
        if selected >= max_tickets:
            break
    print(f"🎫 已選 {selected} 張票")

    # === 勾選服務條款 ===
    # 等 checkbox 可點擊
    checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="person_agree_terms"]')))
    # 滾動到 checkbox
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", checkbox)
    checkbox.click()
    # 觸發 Angular/React 更新
    driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", checkbox)
    time.sleep(1)
    print("✅ 已勾選服務條款")

    # 等待下一步按鈕生成
    timeout = time.time() + 10
    next_btn = None
    while time.time() < timeout:
        next_btn = driver.execute_script("""
            let btns = Array.from(document.querySelectorAll("a.btn, button.btn"));
            for (let b of btns){
                if (b.innerText.includes("下一步") && !b.disabled) return b;
            }
            return null;
        """)
        if next_btn:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
            driver.execute_script("arguments[0].click();", next_btn)
            print("✅ 已點下一步")
            break
        time.sleep(0.2)
    else:
        print("❌ 下一步按鈕一直無法點")

    input("流程完成，按 Enter 關閉瀏覽器...")
except Exception as e:
    print("程式出錯：", e)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    print(f"📄 已輸出 debug_checkbox.html：{save_path}")
    input("按 Enter 關閉瀏覽器...")
finally:
    if 'driver' in locals():
        driver.quit()
