# TemperaturePrediction_MaunaKea
Temperature prediction in Python for the summit of Mauna Kea that is executable on the command line and displays 5 days worth of predictions 
For more information refer to the Project Documentation



## Options:

- `-o, --outname`: Output HTML file for the graph. (Default: "TemperaturePrediction.html")

- `-p, --hours`: Time of day in float from 0-24 to predict. (Default: 18.5)

- `-m, --modelDays`: Number of days to include in the prediction model. (Default: 5)

- `-a, --adjustor`: How much to adjust the model at the start (from 0-1). (Default: 0.7)

## Example

```bash
python your_script_name.py -o MyTemperaturePredictions.html -p 18.5 -m 7 -a 0.8

This command will generate predictions based on 7 days of historical data, adjusting the model with a factor of 0.8, and save the output graph in "MyTemperaturePredictions.html".
