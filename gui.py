import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

from src.emotion_recog.infer import infer_image


class EmotionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("600x700")

        # 이미지 표시용 라벨
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # 감정 결과 표시용 라벨
        self.result_label = tk.Label(root, text="", font=("Arial", 20))
        self.result_label.pack(pady=10)

        # 버튼 영역
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        # 이미지 선택 버튼
        select_btn = tk.Button(btn_frame, text="이미지 선택", command=self.load_image)
        select_btn.pack(side="left", padx=10)

        # 감정 분석 버튼
        analyze_btn = tk.Button(btn_frame, text="감정 분석", command=self.analyze_image)
        analyze_btn.pack(side="left", padx=10)

        self.loaded_image_path = None


    def load_image(self):
        """이미지 선택"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )

        if not file_path:
            return

        self.loaded_image_path = file_path

        # 이미지 로드
        img = Image.open(file_path)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)

        # Label에 이미지 표시
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        self.result_label.config(text="")  # 감정 결과 초기화


    def analyze_image(self):
        """감정 분석"""
        if not self.loaded_image_path:
            messagebox.showwarning("경고", "먼저 이미지를 선택하세요!")
            return

        emotion, score = infer_image(self.loaded_image_path)

        self.result_label.config(text=f"감정: {emotion}  (신뢰도: {score:.2f})",
                                 fg="blue")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
