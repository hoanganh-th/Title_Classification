import pandas as pd
import matplotlib.pyplot as plt

def survey_data(file_path):
    # Load dữ liệu từ file Excel
    df = pd.read_excel(file_path)

    # Cột chứa tiêu đề và nhãn
    title_column = 'News Title'
    category_column = 'Category'

    # Loại bỏ dòng thiếu dữ liệu
    df = df.dropna(subset=[title_column, category_column])

    # Thống kê số lượng tiêu đề theo category
    category_counts = df[category_column].value_counts()
    print("📊 Số lượng tiêu đề theo category:\n", category_counts)

    # Tính độ dài tiêu đề (số từ)
    df['title_length'] = df[title_column].astype(str).apply(lambda x: len(x.split()))

    # Thống kê độ dài tiêu đề
    print("\n📏 Thống kê độ dài tiêu đề (theo số từ):")
    print("Tiêu đề ngắn nhất:", df['title_length'].min())
    print("Tiêu đề dài nhất:", df['title_length'].max())
    print("Độ dài trung bình:", round(df['title_length'].mean(), 2))

    # Thống kê số lượng title ứng với từng độ dài
    length_counts = df['title_length'].value_counts().sort_index()
    print("\n📐 Số lượng tiêu đề theo từng độ dài:")
    print(length_counts)

    # Vẽ biểu đồ phân phối độ dài tiêu đề
    plt.figure(figsize=(10, 6))
    plt.hist(df['title_length'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Phân phối độ dài tiêu đề (số từ)')
    plt.xlabel('Số từ trong tiêu đề')
    plt.ylabel('Số lượng tiêu đề')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    survey_data("News Title.xls")
