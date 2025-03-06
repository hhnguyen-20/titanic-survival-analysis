import dash
from dash import Dash, html, dcc, exceptions
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import joblib
from functools import lru_cache


# Cached data loading
@lru_cache(maxsize=None)
def load_data():
    return pd.read_csv("test.csv")


# Load data and model
test = load_data()
TARGET = 'Survived'
labels = ['0% to <10%', '10% to <20%', '20% to <30%', '30% to <40%', '40% to <50%',
          '50% to <60%', '60% to <70%', '70% to <80%', '80% to <90%', '90% to <100%']
pipeline = joblib.load("model.pkl")

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Titanic Survival Analysis'


# -------------------------
# Helper Functions for Data Filters
# -------------------------
def create_dropdown_options(series):
    """Create dropdown options from a pandas Series."""
    return [{'label': i, 'value': i} for i in sorted(series.unique())]


def create_dropdown_value(series):
    """Return a sorted list of unique values from the Series."""
    return sorted(series.unique().tolist())


def create_slider_marks(values):
    """Create slider marks given a list of values."""
    return {i: {'label': str(i)} for i in values}


def status_in_class(df, klass, status):
    """Return the number of passengers in a class that survived or perished."""
    class_count = 0
    status_count = 0

    for srv, kls in zip(df.Survived, df.Class):
        if kls == klass:  # the class that we want?
            class_count += 1

            if srv == status:  # survived or perished in that class?
                status_count += 1

    return class_count, status_count


# -------------------------
# Helper Functions for Graphs
# -------------------------
def style_layout(title, bgcolor='rgba(255,242,204,100)'):
    """Return a standardized layout for graphs."""
    return dict(
        title=title,
        font_family='Tahoma',
        plot_bgcolor=bgcolor
    )


# -------------------------
# Division 1: Header, Sample Record & Controls
# -------------------------
def create_container1():
    """Creates the header section with title, description, image, and sample record table with controls."""
    return html.Div([
        # Left side: Header, description, and image
        html.Div([
            html.H1("Titanic Survival Analysis"),
            html.P("Explore the Titanic dataset, analyze passenger survival, and predict survival for new passengers."),
            html.Img(src="assets/titanic-sinking.png", className="header-image")
        ], className="container1-left"),

        # Right side: Sample record table with interactive controls
        html.Div([
            dcc.Graph(id='table'),
            html.Div([
                html.Label("Survival status", className='other-labels'),
                daq.BooleanSwitch(id='target-toggle', className='toggle', on=True, color="#FFBD59"),

                html.Label("Sort probability in ascending order", className='other-labels'),
                daq.BooleanSwitch(id='sort-toggle', className='toggle', on=True, color="#FFBD59"),

                html.Label("Number of records", className='other-labels'),
                dcc.Slider(
                    id='n-slider', min=5, max=20, step=1, value=10,
                    marks=create_slider_marks([5, 10, 15, 20])
                ),
            ], id="table-controls", className="table-controls")
        ], className="container1-right")
    ], id="container1", className="container1")


# -------------------------
# Division 2: Data Filters & Graphs
# -------------------------
def create_container2():
    """Creates the data filters and multiple graphs for analysis."""
    return html.Div([
        # Filters section
        html.Div([
            html.H3("Data Filters"),
            html.Label("Passenger Class", className='dropdown-labels'),
            dcc.Dropdown(
                id='class-dropdown',
                className='dropdown',
                multi=True,
                options=create_dropdown_options(test['Class']),
                value=create_dropdown_value(test['Class'])
            ),
            html.Label("Gender", className='dropdown-labels'),
            dcc.Dropdown(
                id='gender-dropdown',
                className='dropdown',
                multi=True,
                options=create_dropdown_options(test['Gender']),
                value=create_dropdown_value(test['Gender'])
            ),
            html.Button(id='update-button', children="Update", n_clicks=0, className="update-button")
        ], id="filters", className="filters"),

        # Graphs section
        html.Div([
            dcc.Loading(
                id="loading-age-histogram",
                type="circle",
                children=dcc.Graph(id="age-histogram")
            ),
            dcc.Loading(
                id="loading-class-barplot",
                type="circle",
                children=dcc.Graph(id="class-barplot")
            ),
            dcc.Loading(
                id="loading-probability-histogram",
                type="circle",
                children=dcc.Graph(id="histogram")
            ),
            dcc.Loading(
                id="loading-survival-barplot",
                type="circle",
                children=dcc.Graph(id="barplot")
            ),
            dcc.Loading(
                id="loading-scatter",
                type="circle",
                children=dcc.Graph(id="age-fare-scatter")
            )
        ], id="graphs", className="graphs")
    ], id="container2", className="container2")


# -------------------------
# Division 3: Predict Survival
# -------------------------
def create_container3():
    """Creates the survival prediction section with a form and result display."""
    return html.Div([
        html.H3("Predict Survival"),
        html.Div([
            html.Label("Passenger Class"),
            dcc.Dropdown(
                id="predict-class",
                options=create_dropdown_options(test['Class']),
                value=create_dropdown_value(test['Class'])[0]
            ),
            html.Label("Gender"),
            dcc.Dropdown(
                id='predict-gender',
                options=[{'label': g, 'value': g} for g in ['Male', 'Female']],
                value='Male'
            ),
            html.Label("Age"),
            dcc.Slider(
                id='predict-age', min=0, max=80, step=1, value=30,
                marks=create_slider_marks([0, 20, 40, 60, 80])
            ),
            html.Label("Siblings/Spouses Aboard"),
            dcc.Slider(
                id='predict-sibsp', min=0, max=8, step=1, value=0,
                marks=create_slider_marks([0, 2, 4, 6, 8])
            ),
            html.Label("Parents/Children Aboard"),
            dcc.Slider(
                id='predict-parch', min=0, max=6, step=1, value=0,
                marks=create_slider_marks([0, 2, 4, 6])
            ),
            html.Label("Fare"),
            dcc.Slider(
                id='predict-fare', min=0, max=500, step=10, value=50,
                marks=create_slider_marks([0, 100, 200, 300, 400, 500])
            ),
            html.Label("Embark Town"),
            dcc.Dropdown(
                id='predict-embark',
                options=create_dropdown_options(test['Embark town']),
                value=create_dropdown_value(test['Embark town'])[0]
            ),
            html.Button(id='predict-button', children="Predict", n_clicks=0, className="predict-button"),
            html.Div(id='predict-result', className="predict-result")
        ], id="predict-form", className="predict-form"),
    ], id="container3", className="container3")


# -------------------------
# App Layout: Combining All Divisions
# -------------------------
app.layout = html.Div(
    id="container",
    children=[
        create_container1(),
        create_container2(),
        create_container3()
    ]
)


# -------------------------
# Callback: Update Visualizations (Graphs and Table)
# -------------------------
@app.callback(
    [Output('age-histogram', 'figure'),
     Output('class-barplot', 'figure'),
     Output('histogram', 'figure'),
     Output('barplot', 'figure'),
     Output('age-fare-scatter', 'figure'),
     Output('table', 'figure'),
     Output('class-dropdown', 'value'),
     Output('gender-dropdown', 'value')],
    [Input('update-button', 'n_clicks'),
     Input('target-toggle', 'on'),
     Input('sort-toggle', 'on'),
     Input('n-slider', 'value')],
    [State('class-dropdown', 'value'),
     State('gender-dropdown', 'value')]
)
def update_output(update_clicks, target, ascending, n,
                  class_value, gender_value):
    """
    Update all graphs and the sample records table based on filter selections.
    """
    dff = test.copy()

    # Apply filters only if the 'Update' button has been clicked at least once
    if update_clicks > 0:
        # Filter by 'Passenger Class' if not empty
        if class_value:
            dff = dff[dff['Class'].isin(class_value)]

        # Filter by 'Gender' if not empty
        if gender_value:
            dff = dff[dff['Gender'].isin(gender_value)]

    # If no data remains after filtering, display empty figures with a warning
    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text="No data available for the selected filters")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, class_value, gender_value

    # --- Age Histogram ---
    age_histogram = px.histogram(dff, x='Age')
    age_histogram.update_layout(**style_layout(f'Distribution of Passenger Ages (n={len(dff)})'))

    # --- Class Barplot ---
    count_1st, survived_1st = status_in_class(dff, 'First', 'Yes')
    count_2nd, survived_2nd = status_in_class(dff, 'Second', 'Yes')
    count_3rd, survived_3rd = status_in_class(dff, 'Third', 'Yes')
    pct_survived_1st = 100 * survived_1st / count_1st if count_1st else 0
    pct_perished_1st = 100 - pct_survived_1st if count_1st else 0
    pct_survived_2nd = 100 * survived_2nd / count_2nd if count_2nd else 0
    pct_perished_2nd = 100 - pct_survived_2nd if count_2nd else 0
    pct_survived_3rd = 100 * survived_3rd / count_3rd if count_3rd else 0
    pct_perished_3rd = 100 - pct_survived_3rd if count_3rd else 0

    class_barplot = go.Figure(data=[
        go.Bar(name='Survived', x=['1st', '2nd', '3rd'],
               y=[pct_survived_1st, pct_survived_2nd, pct_survived_3rd],
               texttemplate='%{y:.2f}%', textposition='auto', marker_color='#3BA27A'),
        go.Bar(name='Perished', x=['1st', '2nd', '3rd'],
               y=[pct_perished_1st, pct_perished_2nd, pct_perished_3rd],
               texttemplate='%{y:.2f}%', textposition='auto', marker_color='#FFBD59')
    ])
    class_barplot.update_layout(**style_layout(f'Passenger Survival Percentage by Class (n={len(dff)})'),
                                barmode='stack')

    # --- Probability Histogram ---
    histogram = px.histogram(
        dff, x='Probability', color=TARGET, marginal="box", nbins=30,
        opacity=0.6, color_discrete_sequence=['#FFBD59', '#3BA27A']
    )
    histogram.update_layout(**style_layout(f'Distribution of Predicted Probabilities (n={len(dff)})'))
    histogram.update_yaxes(title_text="Count")

    # --- Survival Rate Barplot by Binned Probabilities ---
    grouped = dff.groupby('Binned probability', as_index=False, observed=True)['Target'].mean()
    barplot = px.bar(
        grouped, x='Binned probability', y='Target', color_discrete_sequence=['#3BA27A']
    )
    barplot.update_layout(**style_layout(f'Survival Rate by Binned Probabilities (n={len(dff)})'),
                          xaxis={'categoryarray': labels})
    barplot.update_yaxes(title_text="Percentage Survived")

    # --- Age vs Fare Scatter Plot ---
    scatter = px.scatter(
        dff, x='Age', y='Fare', color=TARGET,
        title=f'Age vs Fare by Survival Status (n={len(dff)})', opacity=0.8,
        color_discrete_sequence=['#FFBD59', '#3BA27A']
    )
    scatter.update_layout(font_family='Tahoma')

    # --- Sample Records Table ---
    # Filter table data based on survival toggle
    if target:
        dff_table = dff[dff['Target'] == 1]
    else:
        dff_table = dff[dff['Target'] == 0]

    # When the second toggle is updated or if the slider is updated
    dff_table = dff_table.sort_values('Probability', ascending=ascending).head(n)
    columns = ['Age', 'Gender', 'Class', 'Embark town', TARGET, 'Probability']
    table = go.Figure(data=[go.Table(
        header=dict(
            values=columns,
            fill_color='#23385c',
            line_color='white',
            align='center',
            font=dict(color='white', size=13)
        ),
        cells=dict(
            values=[dff_table[c] for c in columns],
            format=["d", "", "", "", "", ".2%"],
            fill_color=[['white', '#FFF2CC'] * (len(dff_table) - 1)],
            align='center'
        )
    )])
    table.update_layout(title_text=f'Sample Records (n={len(dff_table)})', font_family='Tahoma')

    return age_histogram, class_barplot, histogram, barplot, scatter, table, class_value, gender_value


# -------------------------
# Callback: Predict Survival
# -------------------------
@app.callback(
    Output('predict-result', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('predict-class', 'value'),
     State('predict-gender', 'value'),
     State('predict-age', 'value'),
     State('predict-sibsp', 'value'),
     State('predict-parch', 'value'),
     State('predict-fare', 'value'),
     State('predict-embark', 'value')]
)
def predict_survival(n_clicks, pclass, gender, age, sibsp, parch, fare, embark):
    """
    Predicts survival based on user inputs.
    Validates inputs and returns the prediction and probability.
    """
    if n_clicks == 0:
        return ""

    # Validate inputs
    if None in [age, gender, sibsp, parch, fare, pclass, embark]:
        return html.Div("Please fill all required fields", className="error-message")

    try:
        # Compute extra features
        who = "Child" if age < 18 else ("Man" if gender == "Male" else "Woman")
        adult_male = (gender == "Male" and age >= 18)
        deck = "Missing"
        alone = (sibsp + parch) == 0

        # Prepare input data in expected format
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Sibsp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Class': [pclass],
            'Who': [who],
            'Adult male': [adult_male],
            'Deck': [deck],
            'Embark town': [embark],
            'Alone': [alone]
        })

        # Predict survival and probability using the trained model
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0][1]

        return html.Div([
            html.H4("Prediction:"),
            html.P(f"Survived: {'Yes' if prediction else 'No'}"),
            html.P(f"Probability: {proba * 100:.2f}%")
        ])
    except Exception as e:
        return html.Div([
            html.H4("Prediction Error"),
            html.P(str(e))
        ], className="error-message")


if __name__ == '__main__':
    app.run_server(debug=True)
