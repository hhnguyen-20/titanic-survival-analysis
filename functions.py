########## Helper functions for dropdowns and slider ############
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options

def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value

def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks
