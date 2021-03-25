import pandas as pd


def make():
    df = pd.read_csv("newData.csv", encoding='gbk')
    df.dropna(inplace=True)
    print(df)
    df.to_csv("gbkData.csv", index=False, sep=',', encoding='utf-8')
    df = pd.read_csv("gbkData.csv")  # 先保存又重新读是为了解决编码问题，将编码格式为GBK的源数据转换为UTF-8的数据
    for c in range(0, df.shape[0]):
        if str(df.at[c, "Pos(位置)"]).find('-') != -1:
            df.at[c, "Pos(位置)"] = str(df.at[c, "Pos(位置)"])[:str(df.at[c, "Pos(位置)"]).find("-")]
    pos = df["Pos(位置)"]
    df.drop(["Pos(位置)"], axis=1, inplace=True)
    df.insert(df.shape[1], 'Pos(位置)', pos)
    df.drop(index=[0], inplace=True)
    df.to_csv("data.csv", index=False, sep=',')


if __name__ == '__main__':
    make()
