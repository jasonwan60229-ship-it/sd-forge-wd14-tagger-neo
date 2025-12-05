import os
import gradio as gr
import platform
import sys
import csv
import numpy as np
import gc 
import json
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

# 引入 requests 用於下載模型
try:
    import requests
except ImportError:
    print("WD14 Tagger Neo: 尚未安裝 requests，無法自動下載模型。")
    requests = None

# 引入 WebUI 核心模組
from modules import scripts, script_callbacks, shared, images

# 檢查 onnx runtime
try:
    import onnxruntime as ort
except ImportError:
    print("WD14 Tagger Neo: 尚未安裝 onnxruntime，請確保依賴項已安裝。")
    ort = None

# --- 檔案路徑設定 ---
current_file_path = Path(os.path.abspath(__file__))
EXTENSION_DIR = current_file_path.parent.parent
MODELS_DIR = EXTENSION_DIR / "models"

print(f"WD14 Tagger Neo 偵測到的擴充功能根目錄: {EXTENSION_DIR}")
print(f"WD14 Tagger Neo 偵測到的模型資料夾: {MODELS_DIR}")

# --- 多國語言字典 (I18N) ---
I18N = {
    "Input Image": {
        "中文": "輸入圖片",
        "English": "Input Image",
        "日本語": "入力画像"
    },
    "Select Tagger Model": {
        "中文": "選擇 Tagger 模型",
        "English": "Select Tagger Model",
        "日本語": "Taggerモデルを選択"
    },
    "Threshold": {
        "中文": "標籤閥值 (Threshold)",
        "English": "Tag Threshold",
        "日本語": "タグ閾値"
    },
    "Interrogate": {
        "中文": "開始分析 (Interrogate)",
        "English": "Interrogate",
        "日本語": "解析開始"
    },
    "Unload Model": {
        "中文": "釋放模型 (Unload Model)",
        "English": "Unload Model",
        "日本語": "モデル解放"
    },
    "Output Tags": {
        "中文": "生成的標籤 (Tags)",
        "English": "Generated Tags",
        "日本語": "生成されたタグ"
    },
    "Rating": {
        "中文": "分級 (Rating)",
        "English": "Rating",
        "日本語": "レーティング"
    },
    "Send to Txt2Img": {
        "中文": "傳送到 文生圖 (Txt2Img)",
        "English": "Send to Txt2Img",
        "日本語": "Txt2Imgに送信"
    },
    "Send to Img2Img": {
        "中文": "傳送到 圖生圖 (Img2Img)",
        "English": "Send to Img2Img",
        "日本語": "Img2Imgに送信"
    },
    "Language": {
        "中文": "語言 (Language)",
        "English": "Language",
        "日本語": "言語"
    },
    "Accordion": {
        "中文": "傳送到其他分頁",
        "English": "Send to other tabs",
        "日本語": "他のタブへ送信"
    }
}

# --- 模型配置 ---
MODEL_CONFIGS = {
    "wd-vit-tagger-v3": {
        "repo_id": "SmilingWolf/wd-vit-tagger-v3",
        "onnx_filename": "wd-vit-tagger-v3.onnx", 
        "csv_filename": "wd-vit-tagger-v3.csv", 
        "size": 448
    },
    "wd-convnext-tagger-v3": {
        "repo_id": "SmilingWolf/wd-convnext-tagger-v3",
        "onnx_filename": "wd-convnext-tagger-v3.onnx", 
        "csv_filename": "wd-convnext-tagger-v3.csv", 
        "size": 448
    },
    "wd-swinv2-tagger-v3": {
        "repo_id": "SmilingWolf/wd-swinv2-tagger-v3",
        "onnx_filename": "wd-swinv2-tagger-v3.onnx", 
        "csv_filename": "wd-swinv2-tagger-v3.csv", 
        "size": 448
    },
    "wd-eva02-large-tagger-v3": {
        "repo_id": "SmilingWolf/wd-eva02-large-tagger-v3",
        "onnx_filename": "wd-eva02-large-tagger-v3.onnx", 
        "csv_filename": "wd-eva02-large-tagger-v3.csv", 
        "size": 448
    },
}

class WD14TaggerNeo:
    def __init__(self):
        self.model_loaded = False
        self.current_model_id = None
        self.session = None
        self.tags = {} 
        self.model_configs = MODEL_CONFIGS

    def download_model_files(self, model_id):
        if not requests:
            print("WD14 Tagger Neo 錯誤: 找不到 requests 模組，無法下載模型。")
            return False

        config = self.model_configs[model_id]
        repo_id = config["repo_id"]
        
        files_to_check = [
            ("model.onnx", MODELS_DIR / config["onnx_filename"]),
            ("selected_tags.csv", MODELS_DIR / config["csv_filename"])
        ]
        
        if not MODELS_DIR.exists():
            print(f"建立模型資料夾: {MODELS_DIR}")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for remote_file, local_path in files_to_check:
            if local_path.exists():
                continue 
            
            url = f"https://huggingface.co/{repo_id}/resolve/main/{remote_file}"
            print(f"WD14 Tagger Neo: 正在下載 {local_path.name} ...")
            print(f"來源: {url}")
            
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                print(f"WD14 Tagger Neo: 下載完成 -> {local_path}")
                
            except Exception as e:
                print(f"WD14 Tagger Neo 下載失敗: {e}")
                if local_path.exists():
                    local_path.unlink()
                return False
                
        return True

    def load_tags(self, model_id, csv_filename):
        if model_id in self.tags:
            return True 

        tags_path = MODELS_DIR / csv_filename

        if tags_path.exists():
            try:
                with open(tags_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    model_tags = []
                    for i, row in enumerate(reader):
                        if len(row) > 1:
                            if i == 0 and ("name" in row or "tag" in row[1]):
                                continue
                            model_tags.append(row[1]) 
                
                self.tags[model_id] = model_tags 
                print(f"成功載入模型 {model_id} 的標籤文件，共 {len(model_tags)} 個標籤。")
                return True
            except Exception as e:
                print(f"標籤文件 {csv_filename} 載入失敗: {e}")
                return False
        else:
            print(f"標籤文件缺失：{tags_path}。請確認已下載。")
            return False

    def unload_model(self):
        if self.model_loaded:
            del self.session
            self.session = None
            self.model_loaded = False
            self.current_model_id = None
            gc.collect() 
            return "模型已成功釋放。"
        return "沒有模型處於載入狀態。"

    def load_model(self, model_id):
        if self.current_model_id == model_id and self.model_loaded:
            return f"模型 {model_id} 已載入。"

        if not ort:
            return "錯誤：onnxruntime 未安裝。請檢查您的依賴項。"
        
        print(f"WD14 Tagger Neo: 正在檢查模型檔案 {model_id}...")
        if not self.download_model_files(model_id):
            return f"錯誤：無法下載或找到模型檔案，請檢查網路連線或手動下載。"

        if self.model_loaded:
            self.unload_model()

        config = self.model_configs.get(model_id)
        if not config:
            return f"錯誤：找不到模型 ID: {model_id} 的配置。"
        
        if not self.load_tags(model_id, config["csv_filename"]):
            return "錯誤：模型標籤文件載入失敗。"

        model_path = MODELS_DIR / config["onnx_filename"]

        if not model_path.exists():
            print(f"找不到模型文件：{model_path}")
            return f"錯誤：模型文件({config['onnx_filename']}) 缺失。預期路徑：{model_path.resolve()}。"
        
        print(f"正在載入模型: {model_id} from {model_path}...")
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            
            self.model_loaded = True
            self.current_model_id = model_id
            return f"成功載入模型: {model_id}。"
        except Exception as e:
            print(f"模型載入失敗: {e}")
            self.unload_model() 
            return f"錯誤：模型載入失敗 - {e}"

    def predict(self, image, model_id, threshold=0.35):
        if not image:
            return "請先提供圖片", "---"
        
        load_message = self.load_model(model_id)
        if "錯誤" in load_message:
            return load_message, "---"

        try:
            target_size = self.model_configs[model_id]['size']
            
            if image.mode != 'RGB':
                image = image.convert("RGB")
            
            image = image.resize((target_size, target_size), Image.LANCZOS)
            img_array = np.array(image, dtype=np.float32)
            img_array = img_array[:, :, ::-1] # BGR
            input_tensor = np.expand_dims(img_array, axis=0)
            
        except Exception as e:
            return f"錯誤：圖片前處理失敗 - {e}", "---"

        try:
            print(f"正在使用 {model_id} 執行 ONNX 推論...")
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            probs = self.session.run([output_name], {input_name: input_tensor})[0]
            probs = probs.flatten()
        except Exception as e:
            return f"錯誤：ONNX 推論執行失敗 - {e}", "---"

        current_tags = self.tags[model_id]
        
        if len(current_tags) != len(probs):
             return f"錯誤：標籤數量 ({len(current_tags)}) 與模型輸出數量 ({len(probs)}) 不匹配。", "---"
        
        tag_scores = zip(current_tags, probs)
        
        filtered_tags_with_scores = sorted(
            [(tag, score) for tag, score in tag_scores if score >= threshold],
            key=lambda x: x[1], 
            reverse=True
        )

        rating_tags = []
        general_tags = []
        rating_categories = ["general", "sensitive", "questionable", "explicit"] 

        for tag, score in filtered_tags_with_scores:
            if tag in rating_categories:
                rating_tags.append(tag)
            else:
                clean_tag = tag.replace("_", " ")
                general_tags.append(clean_tag)

        tags_str = ", ".join(general_tags)
        
        if rating_tags:
            rating_str = f"Rating: {rating_tags[0].upper()}"
        else:
            rating_str = "Rating: GENERAL"

        print(f"分析完成，找到 {len(general_tags)} 個標籤。")
        
        return tags_str, rating_str

# 實例化邏輯類別
tagger_backend = WD14TaggerNeo()

# --- 傳輸資料處理函式 (回傳 JSON 字串) ---
def get_transfer_data(tags: str, image: Image.Image):
    # 建立字典
    data = {"tags": tags, "image_b64": None}
    
    if image:
        try:
            # 確保是 RGB 模式再存
            if image.mode != 'RGB':
                image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            data["image_b64"] = img_str
        except Exception as e:
            print(f"WD14 Tagger: 圖片轉碼失敗 - {e}")

    # 強制回傳 JSON 字串，避免 Gradio 傳遞錯誤
    return json.dumps(data)

# --- 標籤傳送的 JavaScript 邏輯 (使用 ID 反查索引) ---
def get_send_js_code(target_tab_type):
    """
    target_tab_type: 'txt2img' 或 'img2img'
    使用您指定的 ID: tab_txt2img 和 tab_img2img
    """
    return f"""
        async function(inputVal) {{
            console.log("WD14 Tagger: JS 啟動，目標", '{target_tab_type}');
            
            // --- 1. 資料解析 (解決 SyntaxError) ---
            var data = {{}};
            // Gradio 有時會將資料包在陣列裡
            var rawData = Array.isArray(inputVal) ? inputVal[0] : inputVal;

            if (typeof rawData === 'string') {{
                try {{
                    data = JSON.parse(rawData);
                }} catch (e) {{
                    console.error("WD14 Tagger: JSON 解析失敗", e);
                    // 嘗試直接當作 tags 使用 (如果上游出錯)
                    data = {{ tags: rawData }}; 
                }}
            }} else if (typeof rawData === 'object') {{
                data = rawData;
            }}

            if (!data || !data.tags) {{
                console.error("WD14 Tagger: 無有效標籤資料");
                return [];
            }}

            // --- 2. 定義目標 ID ---
            var targetTabContentId = 'tab_{target_tab_type}'; // 使用您指定的 ID: tab_txt2img 或 tab_img2img
            var tabContent = gradioApp().getElementById(targetTabContentId);

            if (!tabContent) {{
                console.error("WD14 Tagger: 找不到目標分頁 ID: " + targetTabContentId);
                return [];
            }}

            // --- 3. 寫入提示詞 (尋找該 ID 區塊內的第一個 textarea) ---
            var prompt_textarea = tabContent.querySelector('textarea');
            if (prompt_textarea) {{
                prompt_textarea.value = data.tags;
                prompt_textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                prompt_textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
                console.log("WD14 Tagger: 提示詞已寫入");
            }} else {{
                console.error("WD14 Tagger: 在 " + targetTabContentId + " 中找不到 textarea");
            }}

            // --- 4. 寫入圖片 (僅 Img2Img) ---
            if ('{target_tab_type}' === 'img2img' && data.image_b64) {{
                try {{
                    // 尋找該 ID 區塊內的圖片上傳框
                    var image_input = tabContent.querySelector('input[type="file"]');
                    if (image_input) {{
                        const res = await fetch('data:image/png;base64,' + data.image_b64);
                        const blob = await res.blob();
                        const file = new File([blob], "wd14_input.png", {{ type: "image/png" }});
                        const dt = new DataTransfer();
                        dt.items.add(file);
                        image_input.files = dt.files;
                        image_input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        console.log("WD14 Tagger: 圖片已傳送");
                        // 等待圖片載入完成，避免切換分頁過快導致失敗
                        await new Promise(r => setTimeout(r, 300)); 
                    }}
                }} catch (e) {{
                    console.error("WD14 Tagger: 圖片設定失敗", e);
                }}
            }}

            // --- 5. 切換分頁 (關鍵修正：使用 ID 反查按鈕索引) ---
            // 邏輯：找到 tabs 容器 -> 找到所有分頁內容 -> 找出目標 ID 是第幾個 -> 點擊導航列對應的第幾個按鈕
            var tabsContainer = gradioApp().getElementById('tabs');
            if (!tabsContainer) tabsContainer = gradioApp().querySelector('.tabs');

            if (tabsContainer) {{
                // 找出所有分頁內容 (Gradio 中 class 通常為 tabitem)
                // 為了準確，我們直接找 tabsContainer 下的直接子 div
                var allTabDivs = Array.from(tabsContainer.children).filter(
                    node => node.tagName === 'DIV' && node.classList.contains('tabitem')
                );

                // 找出目標 ID 在這些分頁中的索引 (Index)
                var targetIndex = -1;
                for (var i = 0; i < allTabDivs.length; i++) {{
                    if (allTabDivs[i].id === targetTabContentId) {{
                        targetIndex = i;
                        break;
                    }}
                }}

                if (targetIndex !== -1) {{
                    // 找到導航按鈕列
                    var nav = tabsContainer.querySelector('.tab-nav');
                    if (nav) {{
                        var buttons = nav.querySelectorAll('button');
                        if (buttons && buttons[targetIndex]) {{
                            console.log("WD14 Tagger: 點擊導航按鈕 index:", targetIndex);
                            buttons[targetIndex].click();
                        }} else {{
                            console.error("WD14 Tagger: 找不到對應索引的按鈕");
                        }}
                    }} else {{
                        console.error("WD14 Tagger: 找不到 .tab-nav");
                    }}
                }} else {{
                    console.error("WD14 Tagger: 無法計算分頁索引，找不到 ID " + targetTabContentId + " 在 tabs 中的位置");
                }}
            }}

            return [];
        }}
    """

def pass_tags_to_js(tags):
    return tags

# --- 更新語言的函式 ---
def update_interface_language(lang):
    """根據選擇的語言更新所有介面元件"""
    return [
        gr.update(label=I18N["Input Image"][lang]),           
        gr.update(label=I18N["Select Tagger Model"][lang]),   
        gr.update(label=I18N["Threshold"][lang]),             
        gr.update(value=I18N["Interrogate"][lang]),           
        gr.update(value=I18N["Unload Model"][lang]),          
        gr.update(label=I18N["Output Tags"][lang]),           
        gr.update(label=I18N["Rating"][lang]),                
        gr.update(value=I18N["Send to Txt2Img"][lang]),       
        gr.update(value=I18N["Send to Img2Img"][lang]),       
        gr.update(label=I18N["Accordion"][lang])              
    ]

def on_ui_tabs():
    model_choices = list(tagger_backend.model_configs.keys())
    default_lang = "中文"
    
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row():
            gr.Markdown("## WD14 Tagger Neo")
            lang_dropdown = gr.Dropdown(
                choices=["中文", "English", "日本語"],
                value=default_lang,
                label=I18N["Language"][default_lang],
                elem_id="wd14_lang_selector",
                show_label=True
            )

        with gr.Row():
            with gr.Column(variant='panel'):
                input_image = gr.Image(label=I18N["Input Image"][default_lang], type="pil", elem_id="wd14_input_image")
                
                with gr.Row():
                    model_selector = gr.Dropdown(
                        label=I18N["Select Tagger Model"][default_lang],
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        allow_custom_value=False,
                        elem_id="wd14_model_selector"
                    )
                    
                with gr.Row():
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.35, step=0.05, 
                        label=I18N["Threshold"][default_lang]
                    )
                
                with gr.Row():
                    interrogate_btn = gr.Button(I18N["Interrogate"][default_lang], variant='primary', elem_id="wd14_run_btn")
                    unload_btn = gr.Button(I18N["Unload Model"][default_lang])

            with gr.Column(variant='panel'):
                tags_output = gr.Textbox(label=I18N["Output Tags"][default_lang], lines=5, show_copy_button=True, elem_id="wd14_tags_output")
                rating_output = gr.Label(label=I18N["Rating"][default_lang], elem_id="wd14_rating_output")
                
                with gr.Accordion(I18N["Accordion"][default_lang], open=True) as send_accordion:
                    with gr.Row():
                        send_to_txt2img = gr.Button(I18N["Send to Txt2Img"][default_lang])
                        send_to_img2img = gr.Button(I18N["Send to Img2Img"][default_lang])

        # --- 事件綁定 ---
        
        lang_dropdown.change(
            fn=update_interface_language,
            inputs=[lang_dropdown],
            outputs=[
                input_image, model_selector, threshold_slider, interrogate_btn, unload_btn, 
                tags_output, rating_output, send_to_txt2img, send_to_img2img, send_accordion
            ]
        )

        interrogate_btn.click(
            fn=tagger_backend.predict,
            inputs=[input_image, model_selector, threshold_slider],
            outputs=[tags_output, rating_output]
        )

        unload_btn.click(
            fn=tagger_backend.unload_model,
            inputs=[],
            outputs=[tags_output]
        )
        
        # 修正：使用 get_transfer_data，回傳 JSON 字串給 JS 處理
        send_to_txt2img.click(
            fn=get_transfer_data,
            inputs=[tags_output, input_image],
            outputs=[], 
            _js=get_send_js_code('txt2img')
        )

        send_to_img2img.click(
            fn=get_transfer_data,
            inputs=[tags_output, input_image],
            outputs=[], 
            _js=get_send_js_code('img2img')
        )
        
    return [(tagger_interface, "WD14 Tagger Neo", "wd14_tagger_neo_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

class WD14Script(scripts.Script):
    def title(self):
        return "WD14 Tagger Neo (Script Mode)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("WD14 Tagger Neo", open=False):
            with gr.Row():
                run_btn = gr.Button("分析當前圖片")
                info = gr.Markdown("請在生成後使用，或使用獨立的 Tab 介面。")
        return [run_btn]
