import os

def rename_file(directory):
    # 確認目錄是否存在
    if not os.path.isdir(directory):
        print(f"目錄 {directory} 不存在")
        return
    
    # 遍歷目錄下的所有文件
    for filename in os.listdir(directory):
        # 檢查文件是否已 .bin 結尾
        if filename.endswith('.bin'):
            # 建構舊文件名和新文件名
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, filename.replace('.bin', '.muxraw'))
            # 重新命名文件
            os.rename(old_file, new_file)
            print(f"已經 {old_file} 重新命名為 {new_file}")

if __name__ == "__main__":
    # 輸入目錄
    directory = input("請輸入目錄路徑: ")
    rename_file(directory)