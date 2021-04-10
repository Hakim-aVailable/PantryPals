import pandas as pd
import pulp 
 
calorie_req = 2000
fats_req = 500
proteins_req = 500
carbs_req = 1000
 
foods = pd.read_csv('brunei-food.csv', index_col=['name'])
 
decision_var = pulp.LpVariable.dicts("dec_var", (name for name in foods.index), lowBound=0, cat='Binary')
 
model = pulp.LpProblem("Calories_maximise_problem", pulp.LpMaximize)
 
model += pulp.lpSum(
    [decision_var[name] * foods.loc[name, 'total'] for name in foods.index]
)
 
model += pulp.lpSum(foods.loc[name,'total'] * decision_var[name] for name in foods.index) <= calorie_req
model += pulp.lpSum(foods.loc[name,'fats'] * decision_var[name] for name in foods.index) <= fats_req
model += pulp.lpSum(foods.loc[name,'proteins'] * decision_var[name] for name in foods.index) <= proteins_req
model += pulp.lpSum(foods.loc[name,'carbs'] * decision_var[name] for name in foods.index) <= carbs_req
 
model.solve()
pulp.LpStatus[model.status]
 
output = []
for item in decision_var:
    var_output = {
        'Name': item,
        'Select': decision_var[item].varValue,
        'Total': foods.loc[item,'total'],
        'Fat': foods.loc[item,'fats'],
        'Protein': foods.loc[item,'proteins'],
        'Carbs': foods.loc[item,'carbs']
 
    }
    output.append(var_output)
 
output_df = pd.DataFrame(output)
print(output_df[output_df['Select'] == 1.0])
