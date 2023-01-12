# Fama-French-Model-for-Auto-Trading

Our stock selection strategy is characterized by combining subjectivity, quantification and back testing, using subjective judgment of investment direction, using quantitative models to mine value, and then using back testing to further diversify risks.

In the multi-factor stock selection strategy, our team mainly selects the three Fama-French factors, namely the market capitalization size factor, the book-to-market ratio factor and the market risk premium factor.

In terms of data processing, we first obtain the A-share code of a certain industry, use the rise and fall to calculate the stock's daily return rate, and calculate the median of the market value, as well as the 30% and 70% quantiles of the book-to-market value ratio. Finally, a cross-sectional regression is performed on each stock to select the stock with the smallest alpha.

In addition to the multi-factor stock selection strategy, we also use the 5-day moving average, that is, the five-day average stock transaction price, to observe the stock trend and determine the stocks to buy. The ones that show a clear upward trend and continue to rise along the 5-day line have more investment value.