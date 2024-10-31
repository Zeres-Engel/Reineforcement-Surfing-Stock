import matplotlib.pyplot as plt
import numpy as np
def plot_vn30_scores_and_pricechanges(vn30_actions_scores_df, vn30_price_changes_df, filepath):
    vn30_tickets = vn30_price_changes_df.columns.values
    x = vn30_price_changes_df.index.values
    
    """ print(x)
    print(vn30_actions_scores_df.head())
    print(vn30_tickets) """
    fig, axes = plt.subplots(nrows=vn30_tickets.shape[0] * 2, ncols=1, layout='constrained', figsize=(200, 200))

    for i, ticket in enumerate(vn30_tickets):
        vn30_price_changes_series = vn30_price_changes_df[ticket]
        (vn30_price_changes_series[(vn30_price_changes_series > 0.07) | (vn30_price_changes_series < -0.07)]).to_csv(f'./debug/{ticket}plot.csv')

        axes[2*i].plot(vn30_actions_scores_df[ticket].values, label='actions_scores')
        axes[2*i].legend(loc='upper right')
        axes[2*i].set_title(f'{ticket} actions_scores')

        axes[2*i + 1].plot(vn30_price_changes_df[ticket].values, label='prices_changes')
        axes[2*i + 1].legend(loc='upper right')
        axes[2*i + 1].set_title(f'{ticket} prices_changes')

    
    fig.savefig(filepath, format='png')
    plt.close(fig)
    exit()
    
        

