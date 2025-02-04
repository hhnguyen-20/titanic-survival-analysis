from dash import Dash, html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

from data_prediction import test, TARGET, labels

from functions import create_dropdown_options, create_dropdown_value, create_slider_marks

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Titanic predictions"),
        html.P("Summary of predicted probabilities for Titanic test dataset."),
        html.Img(src="assets/titanic.png"),
        html.Label("Passenger class", className='dropdown-labels'),
        dcc.Dropdown(
            id='class-dropdown', className='dropdown', multi=True,
            options=create_dropdown_options(test['Class']),
            value=create_dropdown_value(test['Class'])),
        html.Label("Gender", className='dropdown-labels'),
        dcc.Dropdown(
            id='gender-dropdown', className='dropdown', multi=True,
            options=create_dropdown_options(test['Gender']),
            value=create_dropdown_value(test['Gender'])),
        html.Button(id='update-button', children="Update", n_clicks=0) # initialize n_clicks=0
        ], id='left-container'),
    html.Div([
        html.Div([
            dcc.Graph(id="histogram"),
            dcc.Graph(id="barplot")
        ], id='visualisation'),
        html.Div([
            dcc.Graph(id='table'),
            html.Div([
                html.Label("Survival status",
                           className='other-labels'),
                daq.BooleanSwitch(id='target_toggle',
                                  className='toggle',
                                  on=True, color="#FFBD59"),
                html.Label("Sort probability in ascending order",
                           className='other-labels'),
                daq.BooleanSwitch(id='sort_toggle',
                                  className='toggle',
                                  on=True, color="#FFBD59"),
                html.Label("Number of records",
                           className='other-labels'),
                dcc.Slider(id='n-slider', min=5, max=20,
                           step=1, value=10,
                           marks=create_slider_marks([5, 10,
                                                      15, 20])),
            ], id='table-side'),
        ], id='data-extract')
    ], id='right-container')
], id='container')


@app.callback(
    [Output(component_id='histogram', component_property='figure'),
     Output(component_id='barplot', component_property='figure'),
     Output(component_id='table', component_property='figure')],
    [State(component_id='class-dropdown', component_property='value'),
     State(component_id='gender-dropdown', component_property='value'),
     Input(component_id='update-button', component_property='n_clicks'),
     Input(component_id='target_toggle', component_property='on'),
     Input(component_id='sort_toggle', component_property='on'),
     Input(component_id='n-slider', component_property='value')]
)
def update_output(class_value, gender_value, n_clicks, target, ascending, n):
    # Update data to dropdown values without overwriting test
    dff = test.copy()

    # Apply filters only if the 'Update' button has been clicked at least once
    if n_clicks > 0:
        # Filter by 'Passenger Class' if not empty
        if class_value:
            dff = dff[dff['Class'].isin(class_value)]

        # Filter by 'Gender' if not empty
        if gender_value:
            dff = dff[dff['Gender'].isin(gender_value)]

    # If no valid data is left after filtering, prevent update to avoid errors
    if dff.empty:
        raise dash.exceptions.PreventUpdate

    # Visual 1: Histogram
    histogram = px.histogram(dff, x='Probability', color=TARGET, marginal="box", nbins=30,
                             opacity=0.6, color_discrete_sequence=['#FFBD59', '#3BA27A'])
    histogram.update_layout(title_text=f'Distribution of probabilities by class (n={len(dff)})',
                            font_family='Tahoma', plot_bgcolor='rgba(255,242,204,100)')
    histogram.update_yaxes(title_text="Count")

    # Visual 2: Barplot
    barplot = px.bar(dff.groupby('Binned probability', as_index=False, observed=True)['Target'].mean(),
                     x='Binned probability', y='Target', color_discrete_sequence=['#3BA27A'])
    barplot.update_layout(title_text=f'Survival rate by binned probabilities (n={len(dff)})',
                          font_family='Tahoma', xaxis={'categoryarray': labels},
                          plot_bgcolor='rgba(255,242,204,100)')
    barplot.update_yaxes(title_text="Percentage survived")

    # Visual 3: Table
    ## When the first toggle is updated
    if target:
        dff = dff[dff['Target'] == 1]
    else:
        dff = dff[dff['Target'] == 0]

    ## When the second toggle is updated or if the slider is updated
    dff = dff.sort_values('Probability', ascending=ascending).head(n)

    columns = ['Age', 'Gender', 'Class', 'Embark town', TARGET, 'Probability']
    table = go.Figure(data=[go.Table(
        header=dict(values=columns, fill_color='#23385c',
                    line_color='white', align='center',
                    font=dict(color='white', size=13)),
        cells=dict(values=[dff[c] for c in columns],
                   format=["d", "", "", "", "", ".2%"],
                   fill_color=[['white', '#FFF2CC'] * (len(dff) - 1)],
                   align='center'))
    ])
    table.update_layout(title_text=f'Sample records (n={len(dff)})', font_family='Tahoma')

    return histogram, barplot, table


if __name__ == '__main__':
    app.title = 'Titanic prediction'
    app.run_server(debug=True)