import pandas as pd
import os


def main():
    file_path = './data/IBM.csv'
    write_path ='./data/IBM.txt'
    file_name = os.path.basename(file_path)
    stock_name = os.path.splitext(file_name)[0]

    df = pd.read_csv(file_path)
    column_names = []
    for column in df.columns.values.tolist():
        # maybe find a better way
        if column == 'Date':
            column_names.append(f"{column}")
        elif column == 'Adj Close':
            column_names.append(f"{stock_name}.Adjusted")
        else:
            column_names.append(f"{stock_name}.{column}")
    df.columns = column_names
    df.to_csv(path_or_buf=write_path, index=True, sep=' ')

if __name__ == '__main__':
    main()
    