import matplotlib.pyplot as plt


class Graph:

    def __init__(self):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.actors = []


    def append_actor(self, actor):
        self.actors.append(actor)


    def draw_all(self):
        plt_counts = len(self.actors)
        for (idx, actor) in enumerate(self.actors, 1):
            date = actor.env.data["DATE"]
            price = actor.env.data["CLOSE"]
            actions = actor.history["ACTION"]
            action_sz= actor.action_sz
            self.draw_action(date, price, actions, action_sz, idx)

            default_cash = actor.default_cash
            cash = actor.history["CASH"]
            portfolio_value = actor.history["PORTFOLIO_VALUE"]
            stock = actor.history["STOCK_HOLD"]
            self.draw_info(default_cash, cash, portfolio_value, stock, idx)

        plt.show()


    def draw_action(self, x, y, actions, action_sz, idx):
        plt.figure("STOCK {}".format(idx))
        plt.plot(x, y)
        for (idx, action) in enumerate(actions):
            if (action == (action_sz/2)): continue
            color = self.colors[int(action)]
            plt.plot(x[idx], y[idx], color + 'o', ms=3)


    def draw_info(self, default_cash, cash, portfolio_value, stock, idx):
        plt.figure("STOCK {}' Info".format(idx))
        plt.subplot(2, 1, 1)
        plt.axhline(y=default_cash, color='lightgray', linestyle="--", label="Inital Possess Money")
        plt.plot(cash, label="Possess Money")
        plt.plot(portfolio_value, label="Possess Money + Stock Value")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(stock, label="Hold Stock Count")
        plt.legend()