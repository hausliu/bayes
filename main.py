import pandas as pd


def make():
    df = pd.read_csv("initData.csv", encoding='gbk')
    df.dropna(inplace=True)
    print(df)
    df.to_csv("gbkData.csv", index=False, sep=',')
    df = pd.read_csv("gbkData.csv")  # 先保存又重新读是为了解决编码问题，将编码格式为GBK的源数据转换为UTF-8的数据
    for c in range(0, df.shape[0]):
        if str(df.at[c, "Pos(位置)"]).find('-') != -1:
            df.at[c, "Pos(位置)"] = str(df.at[c, "Pos(位置)"])[:str(df.at[c, "Pos(位置)"]).find("-")]
    pos = df["Pos(位置)"]
    df.drop(["Player", "Pos(位置)", "Age(年龄)", "Tm(球队)"], axis=1, inplace=True)
    df.insert(df.shape[1], 'Pos(位置)', pos)
    df.to_csv("data.csv", index=False, sep=',')
    with open("data.csv") as f:
        for lines in f:
            lines = lines.replace('C', '0').replace('SF', '1').replace('SG', '2').replace('PG', '3').replace('PF', '4')
            with open('data_01.csv', 'a') as a:
                a.write(lines)


if __name__ == '__main__':
    make()
