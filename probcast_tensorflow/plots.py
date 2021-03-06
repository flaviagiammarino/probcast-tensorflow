import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df, quantiles, targets):

    '''
    Plot the target time series and the predicted quantiles.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with target time series and predicted quantiles.

    quantiles: np.array.
        Quantiles.

    targets: int.
        Number of target time series.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of target time series and predicted quantiles, one subplot for each target.
    '''

    fig = make_subplots(
        subplot_titles=['Target ' + str(i + 1) for i in range(targets)],
        vertical_spacing=0.15,
        rows=targets,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=40, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
        legend=dict(
            traceorder='reversed',
            font=dict(
                color='#000000',
            ),
        ),
    )

    fig.update_annotations(
        font=dict(
            size=13
        )
    )

    for i in range(targets):

        for j in range(len(quantiles) // 2):

            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[- (j + 1)])],
                    name='q' + str(quantiles[j]) + ' - q' + str(quantiles[- (j + 1)]),
                    legendgroup='q' + str(quantiles[j]) + ' - q' + str(quantiles[- (j + 1)]),
                    showlegend=False,
                    mode='lines',
                    line=dict(
                        color='rgba(5, 80, 174, ' + str(0.1 * (j + 1))  + ')',
                        width=0.1
                    )
                ),
                row=i + 1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[j])],
                    name='q' + str(quantiles[j]) + ' - q' + str(quantiles[- (j + 1)]),
                    legendgroup='q' + str(quantiles[j]) + ' - q' + str(quantiles[- (j + 1)]),
                    showlegend=True if i == 0 else False,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(5, 80, 174, ' + str(0.1 * (j + 1)) + ')',
                    line=dict(
                        color='rgba(5, 80, 174, ' + str(0.1 * (j + 1)) + ')',
                        width=0.1,
                    ),
                ),
                row=i + 1,
                col=1
            )

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['target_' + str(i + 1) + '_0.5'],
                name='Median',
                legendgroup='Median',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    width=1,
                    color='rgba(5, 80, 174, 0.5)',
                ),
            ),
            row=i + 1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['target_' + str(i + 1)],
                name='Actual',
                legendgroup='Actual',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    color='#b3b3b3',
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )

        fig.update_xaxes(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )

    return fig
