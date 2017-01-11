# -*- coding: utf-8 -*-
"""
This is a simple implementation of "target shuffling" which was first Introduced
by John Elder in a paper that can be found here http://www.elderresearch.com/target-shuffling

@author: Colin
"""

from scipy import stats
import matplotlib.pyplot as plt
import math

class Target_Shuffle():
    
    def __init__(self):
        pass
    
    def shuffle(self, df, target):
        """
        Shuffles the target column for DataFrame df.
        """
        df[target] = df[target].sample(frac=1).reset_index(drop=True)
        
        return df
        
    def fit_model(self, x, y):
        """
        Returns linear regression model that's fit to x and y.
        """
        return stats.linregress(x, y)
        
    def stat(self, model):
        """
        Returns r-squared value.
        """
        return model.rvalue**2
    
    def calculate_significance(self, orig_stat, r_values):
        """
        Returns estimated p-value for if orig_stat is greater than the array r_values.
        """
        return 1 - len([rval for rval in r_values if rval < orig_stat])/len(r_values)
        
    def test_significance(self, k, df, x_label, y_label, hist=True):
        """
        Tests significance between two variables with r-squared value from linear regression.
        
        Parameters:
            k - Number of tests to run
            df - DataFrame with only two columns
            x_label - Label for x-column in df
            y_label - Label for y-column in df
            hist - If False a histogram won't be plotted, default is True
        Returns:
            estimated p-value for original relationship
        """
        
        orig_model = self.fit_model(df[x_label].values, df[y_label].values)
        orig_stat = self.stat(orig_model)
        r_values = []
        for i in range(k):
            shuffled_df = self.shuffle(df, y_label)
            y = shuffled_df[y_label].values
            x = shuffled_df[x_label].values
            model = self.fit_model(x, y)
            r_values.append(self.stat(model))
        if hist == True:    
            (n, bins, patches) = plt.hist(r_values, bins=20)
            plt.xlim([0, max([orig_stat, max(bins)])+.01])
            plt.axvline(x=orig_stat, color='red', linestyle='--')
            plt.text(orig_stat, max(n), 'Original r-square ({})'.format(round(orig_stat, 2)),
                     rotation=270)
            plt.xlabel('r-squared value')
            plt.ylabel('Frequency')
            plt.show()
        est_pval = self.calculate_significance(orig_stat, r_values)
        print('The estimated p-value for relationship between {} and {} is {}'.format(x_label,
               y_label, est_pval))
        
        return est_pval