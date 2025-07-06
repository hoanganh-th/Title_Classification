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

    # Biểu đồ tròn – Tỷ lệ % từng category
    plt.figure(figsize=(8, 8))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Tỷ lệ tiêu đề theo Category')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Biểu đồ cột – Số lượng tiêu đề theo category
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Số lượng tiêu đề theo Category')
    plt.xlabel('Category')
    plt.ylabel('Số lượng tiêu đề')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Tính độ dài tiêu đề (số từ)
    df['title_length'] = df[title_column].astype(str).apply(lambda x: len(x.split()))

    # Tiêu đề ngắn nhất, dài nhất và độ dài trung bình
    min_len = df['title_length'].min()
    max_len = df['title_length'].max()
    avg_len = round(df['title_length'].mean(), 2)

    shortest_title = df[df['title_length'] == min_len][title_column].values[0]
    longest_title = df[df['title_length'] == max_len][title_column].values[0]

    print("\n📏 Thống kê độ dài tiêu đề:")
    print(f"Tiêu đề ngắn nhất ({min_len} từ): {shortest_title}")
    print(f"Tiêu đề dài nhất ({max_len} từ): {longest_title}")
    print(f"Độ dài trung bình: {avg_len} từ")

    # Biểu đồ histogram – Phân phối độ dài tiêu đề
    plt.figure(figsize=(10, 6))
    plt.hist(df['title_length'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Phân phối độ dài tiêu đề (số từ)')
    plt.xlabel('Số từ trong tiêu đề')
    plt.ylabel('Số lượng tiêu đề')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Biểu đồ cột – Số lượng tiêu đề theo từng độ dài
    length_counts = df['title_length'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    length_counts.plot(kind='bar', color='purple', edgecolor='black')
    plt.title('Số lượng tiêu đề theo độ dài (số từ)')
    plt.xlabel('Độ dài tiêu đề (số từ)')
    plt.ylabel('Số lượng tiêu đề')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    survey_data("News Title.xls")
