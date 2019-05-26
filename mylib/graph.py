import matplotlib.pyplot as plt


def draw_stock_predict(price, action, block=True):
    price = [_price for (_, _price) in price]
    buy_point = []
    buy_price = []
    sell_point = []
    sell_price = []
    for (index, action) in enumerate(action):
        if(action == 1):
            buy_point.append(index)
            buy_price.append(price[index])
        elif(action == 2):
            sell_point.append(index)
            sell_price.append(price[index])
    plt.figure("Prediction")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(price, label="PRICE")
    plt.plot(buy_point, buy_price, 'ro', ms=3, label="BUY")
    plt.plot(sell_point, sell_price, 'bo', ms=3, label="SELL")
    plt.legend()
    plt.show(block=block)


def draw_info(money, property, stock_count, block=True):
    plt.figure("Info")
    plt.subplot(2, 1, 1)
    plt.axhline(y=money[0], color='lightgray', linestyle="--", label="Inital Possess Money")
    plt.plot(money, label="Possess Money")
    plt.plot(property, label="Possess Money + Stock Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(stock_count, label="Hold Stock Count")
    plt.legend()
    plt.show(block=block)