import math
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Backtester:

  def __init__(self, start_date, end_date, strategy) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = start_date
    self.end_date : datetime = end_date
  
    self.user_strategy = strategy
    self.orders : pd.DataFrame = self.user_strategy.generate_orders()
    self.orders["day"] = self.orders["datetime"].apply(lambda x: x.split("T")[0])
    self.orders["hour"] = self.orders["datetime"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[0]))
    self.orders["minute"] = self.orders["datetime"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[1]))
    self.orders["expiration_date"] = self.orders["option_symbol"].apply(lambda x: self.get_expiration_date(x))
    self.orders["sort_by"] = pd.to_datetime(self.orders["datetime"])
    self.orders = self.orders.sort_values(by="sort_by", kind="stable")

    self.options : pd.DataFrame = pd.read_csv("data/cleaned_options_data.csv")
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])
    self.options["hour"] = self.options["ts_recv"].apply(lambda x: int(x.split("T")[1].split(".")[0].split(":")[0])) # FIX

    self.underlying = pd.read_csv("data/spx_minute_level_data_jan_mar_2024.csv")
    self.underlying.columns = self.underlying.columns.str.lower()
    self.underlying["date"] = self.underlying["date"].astype(str)
    self.underlying["day"] = self.underlying["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    self.underlying["hour"] = self.underlying["ms_of_day"].apply(lambda x : self.convert_ms_to_hhmm(x)[0])
    self.underlying["minute"] = self.underlying["ms_of_day"].apply(lambda x : self.convert_ms_to_hhmm(x)[1])
    # self.underlying["day"] = self.underlying["date"].apply(lambda x : x.split(" ")[0])
    # self.underlying["hour"] = self.underlying["date"].apply(lambda x : int(x.split(" ")[1].split("-")[0].split(":")[0]))

    self.pnl : List = []
    self.max_drawdown : float = float("-inf")
    self.overall_return : float = 0
    self.sharpe_ratio : float = 0
    self.overall_score : float = 0
    self.open_orders : pd.DataFrame = pd.DataFrame(columns=["day", "datetime", "option_symbol", 
                                                            "action", "order_size", "expiration_date", 
                                                            "hour", "minute", "bid_px_00", "ask_px_00",
                                                            "running_bid_px_00", "running_ask_px_00"])
    self.open_orders["order_size"] = self.open_orders["order_size"].astype(float)

  def convert_ms_to_hhmm(self, milliseconds):
    total_seconds = milliseconds // 1000
    total_minutes = total_seconds // 60
    hours = total_minutes // 60
    remaining_minutes = total_minutes % 60
    return [hours + 5, remaining_minutes] # + 5 to account for UTC->EST

  def get_expiration_date(self, symbol) -> str:
    numbers : str = symbol.split(" ")[3]
    date : str = numbers[:6]
    date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
    return date_yymmdd

  def parse_option_symbol(self, symbol) -> List:
    """
    example: SPX   240419C00800000
    """
    numbers : str = symbol.split(" ")[3]
    date : str = numbers[:6]
    date_yymmdd : str = "20" + date[0:2] + "-" + date[2:4] + "-" + date[4:6]
    action : str = numbers[6]
    strike_price : float = float(numbers[7:]) / 1000
    return [datetime.strptime(date_yymmdd, "%Y-%m-%d"), action, strike_price]
  
  def check_option_is_open(self, row: pd.Series) -> bool:
    same: pd.DataFrame = self.open_orders[(self.open_orders["option_symbol"] == row["option_symbol"])]
    if len(same) > 0:
      assert len(same) == 1
      assert float(row["order_size"])
      same_index: int = same.index[0]
      if row["action"] == same["action"].iloc[0]:
        self.open_orders.loc[same_index, "order_size"] += float(row["order_size"])
      else:
        if row["order_size"] > same["order_size"].iloc[0]:
          self.open_orders.loc[same_index, "action"] = "B" if row["action"] == "S" else "S"
          self.open_orders.loc[same_index, "order_size"] = float(row["order_size"] - same["order_size"].iloc[0])
        elif row["order_size"] == same["order_size"].iloc[0]:
          self.open_orders = self.open_orders.drop(index=same_index)
        else:
          self.open_orders.loc[same_index, "order_size"] -= float(row["order_size"])
      return True
    return False

  def calculate_pnl(self):
    delta: timedelta = timedelta(days=1)
    current_date: datetime = self.start_date

    while current_date <= self.end_date:
      day_str = str(current_date).split(" ")[0]
      today_orders = self.orders[self.orders["day"] == day_str]
      option_metadata_cache = {}

      for _, row in today_orders.iterrows():
        option_symbol = row["option_symbol"]
        if option_symbol not in option_metadata_cache:
          option_metadata_cache[option_symbol] = self.parse_option_symbol(option_symbol)
        option_metadata = option_metadata_cache[option_symbol]
        order_size = float(row["order_size"])
        strike_price = option_metadata[2]

        matching_row = self.options[
          (self.options["symbol"] == option_symbol) &
          (self.options["ts_recv"] == row["datetime"])
        ]

        if not matching_row.empty:
          matching_row = matching_row.iloc[0]
        else:
          continue

        row["hour"] = 14 if row["hour"] < 14 else min(row["hour"], 21)
        if row["hour"] == 14:
          row["minute"] = 31
        elif row["hour"] == 21:
          row["minute"] = 0

        ask_price = float(matching_row["ask_px_00"])
        buy_price = float(matching_row["bid_px_00"])
        ask_size = float(matching_row["ask_sz_00"])
        buy_size = float(matching_row["bid_sz_00"])

        row["bid_px_00"] = buy_price
        row["ask_px_00"] = ask_price
        row["running_bid_px_00"] = buy_price
        row["running_ask_px_00"] = ask_price

        price = float(self.underlying[
                        (self.underlying["day"] == row["day"]) &
                        (self.underlying["hour"] == row["hour"]) &
                        (self.underlying["minute"] == row["minute"])
                        ]["price"].iloc[0])
                  
        if order_size < 0:
          raise ValueError("Order size must be positive")

        # if (row["action"] == "B" and order_size > ask_size) or (row["action"] == "S" and order_size > buy_size):
        #   raise ValueError(f"Order size exceeds available size; order size: {order_size}, ask size: {ask_size}, buy size: {buy_size}; action: {row['action']}")

        if row["action"] == "B":
          options_cost = order_size * 100 * ask_price
          margin = options_cost + 0.1 * strike_price if option_metadata[1] == "C" else options_cost + 0.1 * price
          margin = 0 if row["option_symbol"] in self.open_orders["option_symbol"].tolist() else margin
          if self.capital >= margin and (self.capital - options_cost + 0.5 > 0):
            self.capital -= options_cost + 0.5
            self.portfolio_value += options_cost
            if not self.check_option_is_open(row):
              new_row = pd.DataFrame([row]).dropna(axis=1, how="all")
              self.open_orders = pd.concat([self.open_orders, new_row], ignore_index=True)
        elif row["action"] == "S":
          options_cost = order_size * 100 * buy_price
          margin = options_cost + 0.1 * strike_price if option_metadata[1] == "C" else options_cost + 0.1 * price
          # already_existing = self.open_orders[self.open_orders["option_symbol"] == row["option_symbol"]]
          # if len(already_existing) > 0:

          if self.capital >= margin:
            self.capital += order_size * buy_price * 100

            if not self.check_option_is_open(row):
              new_row = pd.DataFrame([row]).dropna(axis=1, how="all")
              self.open_orders = pd.concat([self.open_orders, new_row], ignore_index=True)

      for _, order in self.open_orders.iterrows():
        option_metadata = self.parse_option_symbol(order["option_symbol"])
        if str(order["expiration_date"]) == day_str:
          order["hour"] = 14 if order["hour"] < 14 else min(row["hour"], 21)
          if order["hour"] == 14:
            order["minute"] = 31
          elif order["hour"] == 21:
            order["minute"] = 0

          underlying_price = float(self.underlying[
            (self.underlying["day"] == order["expiration_date"]) &
            (self.underlying["hour"] == order["hour"]) &
            (self.underlying["minute"] == order["minute"])
          ]["price"].iloc[0])
          put_call = option_metadata[1]
          strike_price = option_metadata[2]
          order_size = float(order["order_size"])
          underlying_cost = strike_price * 100 * order_size

          if order["action"] == "B":
            if put_call == "C":
              if underlying_price > strike_price:
                stock_value = 100 * order_size * underlying_price
                cost_to_buy = 100 * order_size * strike_price
                self.capital += stock_value - cost_to_buy
                self.portfolio_value -= order["order_size"] * 100 * order["running_ask_px_00"]
            else:
              if underlying_price < strike_price:
                stock_value = 100 * order_size * underlying_price
                cost_to_buy = 100 * order_size * strike_price
                self.capital += (cost_to_buy - stock_value)
                self.portfolio_value -= order["order_size"] * 100 * order["running_ask_px_00"]
          elif order["action"] == "S":
            if put_call == "C":
              if underlying_price > strike_price:
                loss = order_size * 100 * (underlying_price - strike_price)
                self.capital -= loss
            else:
              if underlying_price < strike_price:
                cost = order_size * 100 * (strike_price - underlying_price)
                self.capital -= cost

      # go through open orders and see if price of options people are holding have changed
      for _, order in self.open_orders.iterrows():
        option_symbol = order["option_symbol"]
        order_size = float(order["order_size"])
        
        current_option_data = self.options[
          (self.options["symbol"] == option_symbol) & 
          (self.options["day"] == day_str)
        ]
        
        if not current_option_data.empty:
          current_option_data = current_option_data.iloc[0]
        else:
          continue

        current_bid_price = float(current_option_data["bid_px_00"])
        current_ask_price = float(current_option_data["ask_px_00"])

        if order["action"] == "B":
          original_buy_price = float(order["running_ask_px_00"])
          if current_bid_price > original_buy_price:
            profit = (current_bid_price - original_buy_price) * 100 * order_size
            self.portfolio_value += profit
            
          else:
            loss = (original_buy_price - current_bid_price) * 100 * order_size
            self.portfolio_value -= loss
          order["running_ask_px_00"] = current_bid_price
        elif order["action"] == "S":
          original_sell_price = float(order["running_bid_px_00"])
          if current_ask_price < original_sell_price:
            profit = (original_sell_price - current_ask_price) * 100 * order_size
            self.capital += profit
          else:
            loss = (current_ask_price - original_sell_price) * 100 * order_size
            self.capital -= loss
          order["running_bid_px_00"] = current_ask_price

      # self.portfolio_value = max(self.portfolio_value, 0)
      self.open_orders = self.open_orders[self.open_orders["expiration_date"] != day_str]

      current_date += delta
      self.pnl.append(self.capital + self.portfolio_value)
      print(str(current_date), "capital:", self.capital, "portfolio value:", self.portfolio_value, "total pnl:", (self.capital + self.portfolio_value), "open orders:", len(self.open_orders))

    last_row = self.underlying.iloc[-1]
    for _, order in self.open_orders.iterrows():
      option_metadata = self.parse_option_symbol(order["option_symbol"])
      current_price = last_row["price"] * 100 * order["order_size"]
      if order["action"] == "B":
        self.portfolio_value += 0.9 * current_price
        self.capital -= current_price
      elif order["action"] == "S":
        # self.portfolio_value -= 1.1 * current_price
        self.capital -= 0.1 * current_price

    self.pnl.append(self.capital + self.portfolio_value)
    print("after closing open orders: final capital:", self.capital, "final portfolio value:", self.portfolio_value, "final pnl:", self.pnl[-1])

  def compute_overall_score(self):
    ptr : int = 0
    high_point : float = float("-inf")
    self.max_drawdown = 0.0

    while ptr < len(self.pnl):
      if self.pnl[ptr] > high_point:
        high_point = self.pnl[ptr]
      if self.pnl[ptr] < high_point:
        self.max_drawdown = max(self.max_drawdown, (high_point - self.pnl[ptr]) / high_point)
      ptr += 1

    if self.max_drawdown <= 0:
      self.max_drawdown = 1 * math.pow(10, -10)

    print(f"Max Drawdown: {self.max_drawdown}")

    self.overall_return = 100 * (self.pnl[-1] / 100_000_000)
    print(f"Overall Return: {self.overall_return}%")

    percentage_returns = []
    prev = 100_000_000
    for i in range(len(self.pnl)):
      percentage_returns.append(self.pnl[i] / prev)
      prev = self.pnl[i]

    avg_return = np.sum(percentage_returns) / 61 # 61 trading days in simulation period
    std_return = np.std(percentage_returns)
    
    if std_return > 0.0:
      risk_free_rate = 0.03 / 252
      self.sharpe_ratio = (avg_return - 1 - risk_free_rate) / std_return
      print(f"Sharpe Ratio: {self.sharpe_ratio}")
    else:
      self.sharpe_ratio = 0.0
      print("Sharpe Ratio: Undefined (Standard Deviation = 0)")

    self.overall_score = (self.overall_return / self.max_drawdown) * self.sharpe_ratio
    print(f"Overall Score: {self.overall_score}")

  def plot_pnl(self):
    if not isinstance(self.pnl, list) or len(self.pnl) == 0:
      print("PNL data is not available or empty")

    plt.figure(figsize=(10, 6))
    plt.plot(self.pnl, label="PnL", color="blue", marker="o", linestyle="-", markersize=5)
    
    plt.title("Profit and Loss Over Time", fontsize=14)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("PnL (Profit/Loss)", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()