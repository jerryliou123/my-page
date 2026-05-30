from pdf2docx import Converter
import PySimpleGUI as sg

# 需要 python 版本支援 pdf2docx
def pdf2word(file_path):
    file_name = file_path.split('.')[0]
    doc_file = f'{file_name}.docx'
    p2w = Converter(file_path)
    p2w.convert(doc_file, start=0, end=None)
    p2w.close()
    return doc_file

def main():
    # 選擇主題
    sg.theme('LightBlue5')
    # 設置 window
    layout = [
        [sg.Text('pdfToword', font=('black', 12)),
         sg.Text('', key='filename', size=(50, 1), font=('black', 10), text_color='blue')],
        [sg.Output(size=(80, 10), font=('black', 10))],
        [sg.FilesBrowse('選擇文件', key='file', target='filename'), sg.Button('開始轉換'), sg.Button('退出')]]
    # 创建窗口
    window = sg.Window("Python 與數據分析", layout, font=("black", 15), default_element_size=(50, 1))

    # 事件循环
    while True:
        # 窗口的讀取，有兩個 return（1.事件；2.值）
        event, values = window.read()
        print(event, values)

        if event == "開始轉換":
            # 單個文件
            if values['file'] and values['file'].split('.')[1] == 'pdf':
                filename = pdf2word(values['file'])
                print('文件個數 ：1')
                print('\n' + '轉換成功！' + '\n')
                print('文件保存位置：', filename)
            # 多個文件
            elif values['file'] and values['file'].split(';')[0].split('.')[1] == 'pdf':
                print('文件個數 ：{}'.format(len(values['file'].split(';'))))
                for f in values['file'].split(';'):
                    filename = pdf2word(f)
                    print('\n' + '轉換成功！' + '\n')
                    print('文件保存位置：', filename)
            else:
                print('請選擇 pdf 格式的文件喔!')
        if event in (None, '退出'):
            break

    window.close()

if __name__ == "__main__":
    main()