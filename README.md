
# 星爆臉產生器
讓你的圖片變得非常快!!! 

## 如何使用?

### 前置作業:

1. **安裝 CMake**：請從 CMake 官方網站下載並安裝，確保在安裝時勾選「Add CMake to system PATH」選項，將 CMake 添加至環境變數中。
2. **安裝依賴套件**：使用以下指令來安裝 `requirements.txt` 中列出的所有前置套件：  
```pip install -r requirements.txt```

### 生成圖片:

1. 將你想要星爆的圖片放入 `input_img` 資料夾中。
2. 執行 `SSF_generator.py` 並輸入圖片名稱（不需副檔名）。
3. 接下來，你就可以看到很快的圖片!

#### 範例成果圖片

![範例成果圖片](result/senpai.jpg)


## Gradio Web App

目前已將星爆臉產生器整合至 Gradio，提供了一個簡單易用的線上平台！  
現在，無需安裝任何軟體，你只需開啟以下連結，即可立即體驗星爆臉產生器：  

👉 [前往星爆臉產生器 - Hugging Face](https://huggingface.co/spaces/Jeremy12106/SSF_generator)  

在這個線上網站中，你只需上傳圖片，經過幾秒處理，就能立刻查看並下載生成後的成果。安全又快速，歡迎試用！   