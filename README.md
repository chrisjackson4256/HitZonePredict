# MLBSprayChartPredict

A machine learning model written in Python which predicts which area of the field an MLB batter will hit the ball to.

### Goal

The goal of this project is to predict where the batter is most likely to hit the ball (zones of the field) in an at-bat given the situation and the pitcher he is facing

### The Data:

We use pitch-by-pitch data scraped from Baseball Savant using the pybaseball package.  Types of data that we extract from the raw data include:

  - the pitcher's repertoire: given that each pitcher has a different arsenal of pitches and each pitch moves differently, we use a cluster analysis to categorize pitch types. In this way, we put each pitcher on the same footing.

  - pitch outcomse stats such as groundball and flyball rates

  - the game situation: the inning (and top/bottom), the number of outs, positions of baserunners, the count, positions of fielders

  - the batter's priors: distribution of batted balls into zones

### Output:

  - probabilities for each zone on the field where the batter can hit the ball

  - contributing factors for each prediction (things the defensive team could use to intervene)
