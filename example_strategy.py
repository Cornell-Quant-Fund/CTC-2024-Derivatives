import random
import pandas as pd
from datetime import datetime

class Strategy:
  
  def __init__(self, start_date, end_date, options_data, underlying) -> None:
    self.capital : float = 100_000_000
    self.portfolio_value : float = 0

    self.start_date : datetime = start_date
    self.end_date : datetime = end_date
  
    self.options : pd.DataFrame = pd.read_csv(options_data)
    self.options["day"] = self.options["ts_recv"].apply(lambda x: x.split("T")[0])

    self.underlying = pd.read_csv(underlying)
    self.underlying.columns = self.underlying.columns.str.lower()

  def generate_orders(self) -> pd.DataFrame:
    orders = []
    num_orders = 200
    
    for _ in range(num_orders):
      row = self.options.sample(n=1).iloc[0]
      action = random.choice(["B", "S"])
      
      if action == "B":
        if int(row["ask_sz_00"]) > 1:
          order_size = random.randint(1, int(row["ask_sz_00"]))
      else:
        if int(row["bid_sz_00"]) > 1:
          order_size = random.randint(1, int(row["bid_sz_00"]))
      
      order = {
        "datetime" : row["ts_recv"],
        "option_symbol" : row["symbol"],
        "action" : action,
        "order_size" : order_size
      }
      orders.append(order)
    
    return pd.DataFrame(orders)