import matplotlib.pyplot as plt


class Graph:

    def __init__(self, dates, price_his, block=True):
        self.x = dates.tolist()
        self.y = price_his.tolist()
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.block = block


    def draw(self):
        plt.plot(self.x, self.y)
        plt.show(block=self.block)


    def date_to_price(self, date):
        idx = self.x.index(date)
        price = self.y[idx]

        return price


    def add_dots_on_line(self, dates, dots_style):
        for date, style in zip(dates, dots_style):
            if(style == 0): continue

            price = self.date_to_price(date)
            style = self.colors[int(style)] + 'o'
            plt.plot(date, price, style, ms=3)



def draw_stock_predict(price, action, block=True):
    price = [y for y in price]
    buy_point = []
    buy_price = []
    sell_point = []
    sell_price = []
    for (index, action) in enumerate(action):
        if (action == 1 or action == 2 or action == 3):
            buy_point.append(index)
            buy_price.append(price[index])
        elif (action == 4 or action == 5 or action == 6):
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