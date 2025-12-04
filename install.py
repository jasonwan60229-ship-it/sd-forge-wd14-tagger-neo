import launch

# 檢查 onnxruntime-gpu，如果沒有則退回到 cpu 版本
if not launch.is_installed("onnxruntime-gpu") and not launch.is_installed("onnxruntime"):
    # 嘗試安裝 GPU 版本，如果失敗會提示用戶
    try:
        launch.run_pip("install onnxruntime-gpu", "onnxruntime-gpu requirements for WD14 Tagger Neo")
    except Exception:
        print("WD14 Tagger Neo: GPU 版本安裝失敗，嘗試安裝 CPU 版本...")
        launch.run_pip("install onnxruntime", "onnxruntime-cpu requirements for WD14 Tagger Neo")

# 確保 requests 庫已安裝，如果未來要實現模型自動下載功能會需要
if not launch.is_installed("requests"):
    launch.run_pip("install requests", "requests library for WD14 Tagger Neo")

# Pillow 已經是 WebUI 的核心依賴，通常不需要單獨安裝。