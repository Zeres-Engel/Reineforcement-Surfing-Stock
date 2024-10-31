import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc, candlestick_ochl
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np

UP_COLOR =  "green"
DOWN_CLOR = "red"

class StockMarketGraph:


    def __init__(self, ticket, render_range, n_predictions) -> None:
        self.render_range = render_range
        self.ticket = ticket
        self.n_predictions = n_predictions

    
    def reset(self, pivot_day_ind):
        # We are using the style ‘ggplot’
        self.pivot_day_ind = pivot_day_ind
        plt.style.use('dark_background') # plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8)) 

        # Create a new axis for score which shares its x-axis with price
        self.score_ax = plt.subplot2grid((9,1), (0,0), rowspan=3, colspan=1) #self.price_ax.twinx() 

        # Create top subplot for price axis
        self.price_ax = plt.subplot2grid((9,1), (3,0), rowspan=5, colspan=1, sharex=self.score_ax)
        
        # Create bottom subplot for volume which shares its x-axis
        self.volume_ax = plt.subplot2grid((9,1), (8,0), rowspan=1, colspan=1, sharex=self.score_ax)
        
        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%Y-%m-%d')
        #self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        
        # Add paddings to make graph easier to view
        plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

    def _render_price(self, ohlc_history, dates):
        self.price_ax.clear()
        candlesticks = zip(dates,
                           ohlc_history['open'], ohlc_history['high'],
                           ohlc_history['low'], ohlc_history['close'])

        candlestick_ohlc(self.price_ax, candlesticks, width=0.8, colorup=UP_COLOR, colordown=DOWN_CLOR, alpha=0.8)

    def _render_volume(self, ohlc_history, volume_history, dates):
        self.volume_ax.clear()

        # Viết kiểu khác
        bullish = ohlc_history['close'] >= ohlc_history['open']
        bearish = ohlc_history['close'] < ohlc_history['open']
        self.volume_ax.bar(dates[bullish], volume_history[bullish], color=UP_COLOR, alpha=0.8, width=0.8, align="center")
        self.volume_ax.bar(dates[bearish], volume_history[bearish], color=DOWN_CLOR, alpha=0.8, width=0.8, align="center")

        # self.volume_ax.fill_between(dates, volume_history, 0)

    def _render_score(self, score_history, dates):
        self.score_ax.clear()
        self.score_ax.plot(dates, score_history, color="blue")

    def render(self, vn30_raw_dict, score_history, trading_dates, curr_day_ind, mode):
        score_history_size = len(score_history)
        assert score_history_size == (curr_day_ind - self.pivot_day_ind), f"Kich co cua score_history {score_history_size} va curr_day_ind {curr_day_ind}"

        raw_stock_df = vn30_raw_dict[self.ticket]
        start_date_ind = max(self.pivot_day_ind, curr_day_ind - self.render_range) 
        end_date_ind = curr_day_ind
        render_dates = trading_dates[start_date_ind: end_date_ind]
        # print(len(render_dates))

        rendered_ohlc_history = raw_stock_df.loc[render_dates, ['open', 'high', 'low', 'close']]
        rendered_volume_history = raw_stock_df.loc[render_dates, "volume"]
        rendered_score_history = score_history[-self.render_range:]
        rendered_mdates = mpl_dates.date2num(render_dates)
        
        """ print(curr_day_ind)
        print(self.pivot_day_ind)
        print(start_date_ind)
        print(end_date_ind)
        print(len(render_dates))
        print(len(rendered_score_history)) """
        
        self._render_price(rendered_ohlc_history, rendered_mdates)
        self._render_volume(rendered_ohlc_history, rendered_volume_history, rendered_mdates)
        self._render_score(rendered_score_history, rendered_mdates)
        
        # we need to set layers every step, because we are clearing subplots every step
        self.fig.suptitle(f"Ticket: {self.ticket}\n Score: {score_history[-1]}")
        self.score_ax.set_xlabel('Date')
        self.score_ax.set_ylabel('Score')
        self.price_ax.set_ylabel('Price')
        self.volume_ax.set_ylabel('Volume')

        # beautify the x-labels (Our Date format)
        self.score_ax.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        if mode == "rgb_array":
            return img
        elif mode == "human":
            # img is rgb, convert to opencv's default bgr
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # display image with OpenCV or any operation you like
            cv2.imshow(f"Stock market graph",image)

            if cv2.waitKey(250) & 0xFF == ord("q"):
                cv2.destroyAllWindows()