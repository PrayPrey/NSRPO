"""
Statistical Significance Testing for NSRPO Evaluation
Task 12: Implement Statistical Significance Testing - T-tests, p-values, confidence intervals

Comprehensive statistical testing framework for academic rigor in model comparison
and evaluation. Includes multiple comparison corrections and effect size calculations.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, wilcoxon, mannwhitneyu, 
    bootstrap, chi2_contingency, fisher_exact,
    shapiro, levene, bartlett
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.power import ttest_power
import pandas as pd
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions_met: bool = True
    power: Optional[float] = None
    sample_size_recommendation: Optional[int] = None


class StatisticalTester:
    """
    Comprehensive statistical testing framework for model evaluation.
    
    Provides rigorous statistical analysis for model comparisons including
    parametric/non-parametric tests, multiple comparison corrections,
    and effect size calculations.
    """
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level (default: 0.05)
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store test results for multiple comparison correction
        self.test_results = []
    
    def test_normality(self, data: np.ndarray, test: str = 'shapiro') -> Dict[str, Any]:
        """
        Test for normality of data distribution.
        
        Args:
            data: Data to test for normality
            test: Test to use ('shapiro', 'ks', 'jarque_bera')
            
        Returns:
            Dictionary with normality test results
        """
        if len(data) < 3:
            return {
                'test_name': test,
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': False,
                'note': 'Insufficient data for normality testing'
            }
        
        if test == 'shapiro':
            if len(data) > 5000:
                # Shapiro-Wilk is not reliable for large samples
                test = 'ks'
        
        try:
            if test == 'shapiro':
                stat, p_value = shapiro(data)
            elif test == 'ks':
                # Kolmogorov-Smirnov test against normal distribution
                stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, np.mean(data), np.std(data)))
            elif test == 'jarque_bera':
                stat, p_value = stats.jarque_bera(data)
            else:
                raise ValueError(f"Unknown normality test: {test}")
            
            is_normal = p_value > self.alpha
            
            return {
                'test_name': test,
                'statistic': float(stat),
                'p_value': float(p_value),
                'is_normal': is_normal,
                'interpretation': f"Data {'appears' if is_normal else 'does not appear'} to be normally distributed"
            }
            
        except Exception as e:
            return {
                'test_name': test,
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': False,
                'error': str(e)
            }
    
    def test_equal_variances(self, *groups: np.ndarray, test: str = 'levene') -> Dict[str, Any]:
        """
        Test for equal variances across groups.
        
        Args:
            *groups: Groups to test for equal variances
            test: Test to use ('levene', 'bartlett')
            
        Returns:
            Dictionary with variance test results
        """
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for variance testing")
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {
                'test_name': test,
                'statistic': np.nan,
                'p_value': np.nan,
                'equal_variances': False,
                'note': 'Insufficient groups with data'
            }
        
        try:
            if test == 'levene':
                stat, p_value = levene(*groups)
            elif test == 'bartlett':
                stat, p_value = bartlett(*groups)
            else:
                raise ValueError(f"Unknown variance test: {test}")
            
            equal_variances = p_value > self.alpha
            
            return {
                'test_name': test,
                'statistic': float(stat),
                'p_value': float(p_value),
                'equal_variances': equal_variances,
                'interpretation': f"Variances {'appear' if equal_variances else 'do not appear'} to be equal"
            }
            
        except Exception as e:
            return {
                'test_name': test,
                'statistic': np.nan,
                'p_value': np.nan,
                'equal_variances': False,
                'error': str(e)
            }
    
    def compare_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
        test_type: str = 'auto',
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Compare two groups using appropriate statistical test.
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether data is paired
            test_type: Type of test ('auto', 'parametric', 'nonparametric')
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            StatisticalTestResult object
        """
        # Data validation
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) == 0 or len(group2) == 0:
            return StatisticalTestResult(
                test_name="insufficient_data",
                statistic=np.nan,
                p_value=1.0,
                interpretation="Insufficient data for comparison"
            )
        
        # Determine appropriate test
        if test_type == 'auto':
            test_type = self._determine_test_type(group1, group2, paired)
        
        try:
            if test_type == 'parametric':
                if paired:
                    # Paired t-test
                    if len(group1) != len(group2):
                        raise ValueError("Paired test requires equal length groups")
                    stat, p_value = ttest_rel(group1, group2, alternative=alternative)
                    test_name = "paired_ttest"
                    effect_size = self._cohens_d_paired(group1, group2)
                else:
                    # Independent t-test
                    # Check equal variances
                    var_test = self.test_equal_variances(group1, group2)
                    equal_var = var_test['equal_variances']
                    
                    stat, p_value = ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
                    test_name = f"independent_ttest_{'equal_var' if equal_var else 'unequal_var'}"
                    effect_size = self._cohens_d_independent(group1, group2)
            
            else:  # nonparametric
                if paired:
                    # Wilcoxon signed-rank test
                    if len(group1) != len(group2):
                        raise ValueError("Paired test requires equal length groups")
                    stat, p_value = wilcoxon(group1, group2, alternative=alternative)
                    test_name = "wilcoxon_signed_rank"
                    effect_size = self._rank_biserial_correlation_paired(group1, group2)
                else:
                    # Mann-Whitney U test
                    stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
                    test_name = "mann_whitney_u"
                    effect_size = self._rank_biserial_correlation_independent(group1, group2)
            
            # Compute confidence interval
            if test_type == 'parametric':
                ci = self._compute_mean_difference_ci(group1, group2, paired)
            else:
                ci = None  # CI for median difference is more complex
            
            # Compute power analysis
            if test_type == 'parametric':
                power = self._compute_power(group1, group2, effect_size)
                sample_size_rec = self._recommend_sample_size(effect_size)
            else:
                power = None
                sample_size_rec = None
            
            # Interpretation
            significant = p_value < self.alpha
            interpretation = self._interpret_comparison_result(
                significant, p_value, effect_size, test_name
            )
            
            result = StatisticalTestResult(
                test_name=test_name,
                statistic=float(stat),
                p_value=float(p_value),
                effect_size=float(effect_size) if effect_size is not None else None,
                confidence_interval=ci,
                interpretation=interpretation,
                assumptions_met=test_type == 'parametric',
                power=power,
                sample_size_recommendation=sample_size_rec
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            return StatisticalTestResult(
                test_name="error",
                statistic=np.nan,
                p_value=1.0,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def compare_multiple_groups(
        self,
        groups: List[np.ndarray],
        group_names: Optional[List[str]] = None,
        test_type: str = 'auto',
        post_hoc: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple groups using ANOVA or Kruskal-Wallis.
        
        Args:
            groups: List of group data arrays
            group_names: Names for the groups
            test_type: Type of test ('auto', 'parametric', 'nonparametric')
            post_hoc: Whether to perform post-hoc comparisons
            
        Returns:
            Dictionary with results including post-hoc comparisons
        """
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for comparison")
        
        # Clean data
        groups = [np.array(g)[~np.isnan(np.array(g))] for g in groups]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            return {
                'overall_test': 'insufficient_data',
                'p_value': 1.0,
                'interpretation': 'Insufficient data for comparison'
            }
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(groups))]
        
        # Determine test type
        if test_type == 'auto':
            # Check normality and equal variances for all groups
            normality_ok = all(self.test_normality(g)['is_normal'] for g in groups if len(g) >= 3)
            variance_ok = self.test_equal_variances(*groups)['equal_variances']
            test_type = 'parametric' if normality_ok and variance_ok else 'nonparametric'
        
        results = {}
        
        try:
            if test_type == 'parametric':
                # One-way ANOVA
                stat, p_value = stats.f_oneway(*groups)
                test_name = "one_way_anova"
                effect_size = self._eta_squared(groups, stat, p_value)
            else:
                # Kruskal-Wallis test
                stat, p_value = stats.kruskal(*groups)
                test_name = "kruskal_wallis"
                effect_size = self._epsilon_squared(groups, stat)
            
            results['overall_test'] = {
                'test_name': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'effect_size': float(effect_size) if effect_size is not None else None,
                'significant': p_value < self.alpha,
                'interpretation': f"Groups {'significantly' if p_value < self.alpha else 'not significantly'} differ"
            }
            
            # Post-hoc comparisons if overall test is significant
            if post_hoc and p_value < self.alpha:
                post_hoc_results = {}
                p_values = []
                
                for i, group1 in enumerate(groups):
                    for j, group2 in enumerate(groups[i+1:], i+1):
                        comparison_name = f"{group_names[i]}_vs_{group_names[j]}"
                        
                        # Use same test type as overall test
                        if test_type == 'parametric':
                            stat_ph, p_val_ph = ttest_ind(group1, group2)
                            effect_size_ph = self._cohens_d_independent(group1, group2)
                        else:
                            stat_ph, p_val_ph = mannwhitneyu(group1, group2)
                            effect_size_ph = self._rank_biserial_correlation_independent(group1, group2)
                        
                        post_hoc_results[comparison_name] = {
                            'statistic': float(stat_ph),
                            'p_value_uncorrected': float(p_val_ph),
                            'effect_size': float(effect_size_ph) if effect_size_ph is not None else None
                        }
                        p_values.append(p_val_ph)
                
                # Multiple comparison correction
                if p_values:
                    corrected_results = self.correct_multiple_comparisons(
                        p_values, method='bonferroni'
                    )
                    
                    for idx, (comparison_name, result) in enumerate(post_hoc_results.items()):
                        result['p_value_corrected'] = corrected_results['corrected_p_values'][idx]
                        result['significant_corrected'] = corrected_results['significant'][idx]
                
                results['post_hoc'] = post_hoc_results
            
            return results
            
        except Exception as e:
            return {
                'overall_test': 'error',
                'error': str(e),
                'interpretation': f"Multiple group comparison failed: {str(e)}"
            }
    
    def compare_proportions(
        self,
        successes: List[int],
        totals: List[int],
        group_names: Optional[List[str]] = None,
        test_type: str = 'chi2'
    ) -> Dict[str, Any]:
        """
        Compare proportions between groups.
        
        Args:
            successes: Number of successes in each group
            totals: Total number of trials in each group
            group_names: Names for the groups
            test_type: Test to use ('chi2', 'fisher')
            
        Returns:
            Dictionary with proportion comparison results
        """
        if len(successes) != len(totals):
            raise ValueError("Length of successes and totals must match")
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(successes))]
        
        # Create contingency table
        failures = [total - success for success, total in zip(successes, totals)]
        contingency_table = np.array([successes, failures])
        
        results = {
            'contingency_table': contingency_table.tolist(),
            'group_names': group_names,
            'proportions': [s/t if t > 0 else 0.0 for s, t in zip(successes, totals)]
        }
        
        try:
            if test_type == 'chi2':
                if len(successes) == 2 and totals[0] > 0 and totals[1] > 0:
                    # 2x2 table, can also compute odds ratio
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    # Compute odds ratio and confidence interval
                    odds_ratio, ci_lower, ci_upper = self._odds_ratio_ci(contingency_table)
                    
                    results.update({
                        'test_name': 'chi2_test',
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'odds_ratio': float(odds_ratio) if not np.isnan(odds_ratio) else None,
                        'odds_ratio_ci': [float(ci_lower), float(ci_upper)] if not np.isnan(ci_lower) else None,
                        'expected_frequencies': expected.tolist()
                    })
                else:
                    # Multi-group comparison
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    results.update({
                        'test_name': 'chi2_test_multi',
                        'chi2_statistic': float(chi2_stat),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'expected_frequencies': expected.tolist()
                    })
                
            elif test_type == 'fisher':
                if len(successes) == 2:
                    # Fisher's exact test (2x2 only)
                    odds_ratio, p_value = fisher_exact(contingency_table)
                    results.update({
                        'test_name': 'fisher_exact',
                        'odds_ratio': float(odds_ratio),
                        'p_value': float(p_value)
                    })
                else:
                    raise ValueError("Fisher's exact test only supports 2x2 tables")
            
            # Add interpretation
            significant = results.get('p_value', 1.0) < self.alpha
            results['significant'] = significant
            results['interpretation'] = f"Proportions {'significantly' if significant else 'not significantly'} differ between groups"
            
            return results
            
        except Exception as e:
            results.update({
                'test_name': 'error',
                'error': str(e),
                'interpretation': f"Proportion comparison failed: {str(e)}"
            })
            return results
    
    def mcnemar_test(
        self,
        contingency_table: np.ndarray,
        exact: bool = None
    ) -> Dict[str, Any]:
        """
        Perform McNemar's test for paired proportions.
        
        Args:
            contingency_table: 2x2 contingency table
            exact: Whether to use exact test
            
        Returns:
            Dictionary with McNemar test results
        """
        if contingency_table.shape != (2, 2):
            raise ValueError("McNemar test requires 2x2 contingency table")
        
        try:
            result = mcnemar(contingency_table, exact=exact)
            
            return {
                'test_name': 'mcnemar',
                'statistic': float(result.statistic),
                'p_value': float(result.pvalue),
                'significant': result.pvalue < self.alpha,
                'interpretation': f"Paired proportions {'significantly' if result.pvalue < self.alpha else 'not significantly'} differ"
            }
            
        except Exception as e:
            return {
                'test_name': 'mcnemar_error',
                'error': str(e),
                'interpretation': f"McNemar test failed: {str(e)}"
            }
    
    def correct_multiple_comparisons(
        self,
        p_values: List[float],
        method: str = 'bonferroni',
        alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Correct for multiple comparisons.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha
        
        try:
            rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method=method
            )
            
            return {
                'method': method,
                'original_p_values': p_values,
                'corrected_p_values': corrected_p_values.tolist(),
                'significant': rejected.tolist(),
                'alpha_original': alpha,
                'alpha_bonferroni': float(alpha_bonf),
                'alpha_sidak': float(alpha_sidak),
                'num_significant_original': sum(p < alpha for p in p_values),
                'num_significant_corrected': sum(rejected)
            }
            
        except Exception as e:
            return {
                'method': method,
                'error': str(e),
                'original_p_values': p_values,
                'corrected_p_values': p_values,
                'significant': [p < alpha for p in p_values]
            }
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 10000,
        method: str = 'percentile'
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute statistic
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            method: Bootstrap method ('percentile', 'bca')
            
        Returns:
            Tuple of (lower_bound, upper_bound, additional_info)
        """
        try:
            # Use scipy's bootstrap function
            res = bootstrap(
                (data,), 
                statistic_func, 
                n_resamples=n_bootstrap,
                confidence_level=confidence_level,
                method=method,
                random_state=self.random_state
            )
            
            return (
                float(res.confidence_interval.low),
                float(res.confidence_interval.high),
                {
                    'method': method,
                    'confidence_level': confidence_level,
                    'n_bootstrap': n_bootstrap,
                    'bootstrap_distribution_mean': float(np.mean(res.bootstrap_distribution)),
                    'bootstrap_distribution_std': float(np.std(res.bootstrap_distribution))
                }
            )
            
        except Exception as e:
            # Fallback to simple percentile method
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_samples.append(statistic_func(sample))
            
            bootstrap_samples = np.array(bootstrap_samples)
            alpha = 1 - confidence_level
            lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
            upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
            
            return (
                float(lower),
                float(upper),
                {
                    'method': 'percentile_fallback',
                    'confidence_level': confidence_level,
                    'n_bootstrap': n_bootstrap,
                    'error': str(e)
                }
            )
    
    def _determine_test_type(self, group1: np.ndarray, group2: np.ndarray, paired: bool) -> str:
        """Determine appropriate test type based on assumptions."""
        # Check normality
        norm1 = self.test_normality(group1)['is_normal']
        norm2 = self.test_normality(group2)['is_normal']
        
        # Check equal variances (for independent samples)
        if not paired:
            equal_var = self.test_equal_variances(group1, group2)['equal_variances']
        else:
            equal_var = True  # Not relevant for paired tests
        
        # Decide based on assumptions
        if norm1 and norm2 and equal_var:
            return 'parametric'
        else:
            return 'nonparametric'
    
    def _cohens_d_independent(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d for independent samples."""
        n1, n2 = len(group1), len(group2)
        if n1 <= 1 or n2 <= 1:
            return np.nan
        
        # Pooled standard deviation
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return np.nan
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _cohens_d_paired(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d for paired samples."""
        diff = group1 - group2
        if len(diff) <= 1:
            return np.nan
        
        return np.mean(diff) / np.std(diff, ddof=1)
    
    def _rank_biserial_correlation_independent(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation for independent samples."""
        try:
            U, _ = mannwhitneyu(group1, group2)
            n1, n2 = len(group1), len(group2)
            return 1 - (2 * U) / (n1 * n2)
        except:
            return np.nan
    
    def _rank_biserial_correlation_paired(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate rank-biserial correlation for paired samples."""
        try:
            diff = group1 - group2
            diff_nonzero = diff[diff != 0]
            if len(diff_nonzero) == 0:
                return 0.0
            
            pos = np.sum(diff_nonzero > 0)
            neg = np.sum(diff_nonzero < 0)
            return (pos - neg) / (pos + neg)
        except:
            return np.nan
    
    def _eta_squared(self, groups: List[np.ndarray], f_stat: float, p_value: float) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        try:
            # Total sum of squares
            all_data = np.concatenate(groups)
            ss_total = np.sum((all_data - np.mean(all_data))**2)
            
            # Between-group sum of squares
            ss_between = sum(len(g) * (np.mean(g) - np.mean(all_data))**2 for g in groups)
            
            return ss_between / ss_total if ss_total > 0 else np.nan
        except:
            return np.nan
    
    def _epsilon_squared(self, groups: List[np.ndarray], h_stat: float) -> float:
        """Calculate epsilon-squared effect size for Kruskal-Wallis."""
        try:
            n_total = sum(len(g) for g in groups)
            return (h_stat - len(groups) + 1) / (n_total - len(groups))
        except:
            return np.nan
    
    def _compute_mean_difference_ci(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        paired: bool,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for mean difference."""
        try:
            alpha = 1 - confidence
            
            if paired:
                diff = group1 - group2
                mean_diff = np.mean(diff)
                se_diff = stats.sem(diff)
                df = len(diff) - 1
                
                if df <= 0:
                    return (np.nan, np.nan)
                
                t_crit = stats.t.ppf(1 - alpha/2, df)
                margin = t_crit * se_diff
                
                return (mean_diff - margin, mean_diff + margin)
            else:
                mean1, mean2 = np.mean(group1), np.mean(group2)
                se1, se2 = stats.sem(group1), stats.sem(group2)
                
                mean_diff = mean1 - mean2
                se_diff = np.sqrt(se1**2 + se2**2)
                
                # Use Welch's t-test degrees of freedom
                df = (se1**2 + se2**2)**2 / (se1**4/(len(group1)-1) + se2**4/(len(group2)-1))
                
                if df <= 0:
                    return (np.nan, np.nan)
                
                t_crit = stats.t.ppf(1 - alpha/2, df)
                margin = t_crit * se_diff
                
                return (mean_diff - margin, mean_diff + margin)
                
        except:
            return (np.nan, np.nan)
    
    def _compute_power(self, group1: np.ndarray, group2: np.ndarray, effect_size: float) -> Optional[float]:
        """Compute statistical power for t-test."""
        try:
            if effect_size is None or np.isnan(effect_size):
                return None
            
            n1, n2 = len(group1), len(group2)
            power = ttest_power(effect_size, n1, self.alpha, alternative='two-sided')
            return float(power)
        except:
            return None
    
    def _recommend_sample_size(self, effect_size: float, desired_power: float = 0.8) -> Optional[int]:
        """Recommend sample size for desired power."""
        try:
            if effect_size is None or np.isnan(effect_size) or effect_size == 0:
                return None
            
            from statsmodels.stats.power import ttest_power
            
            # Binary search for required sample size
            n_min, n_max = 5, 10000
            while n_max - n_min > 1:
                n_mid = (n_min + n_max) // 2
                power = ttest_power(effect_size, n_mid, self.alpha)
                
                if power < desired_power:
                    n_min = n_mid
                else:
                    n_max = n_mid
            
            return n_max if n_max < 10000 else None
        except:
            return None
    
    def _odds_ratio_ci(self, contingency_table: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
        """Compute odds ratio and confidence interval."""
        try:
            a, b, c, d = contingency_table.flatten()
            
            if b == 0 or c == 0 or a == 0 or d == 0:
                return (np.nan, np.nan, np.nan)
            
            odds_ratio = (a * d) / (b * c)
            
            # Log odds ratio confidence interval
            log_or = np.log(odds_ratio)
            se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
            
            alpha = 1 - confidence
            z_crit = stats.norm.ppf(1 - alpha/2)
            
            ci_lower = np.exp(log_or - z_crit * se_log_or)
            ci_upper = np.exp(log_or + z_crit * se_log_or)
            
            return (odds_ratio, ci_lower, ci_upper)
        except:
            return (np.nan, np.nan, np.nan)
    
    def _interpret_comparison_result(
        self, 
        significant: bool, 
        p_value: float, 
        effect_size: Optional[float],
        test_name: str
    ) -> str:
        """Generate interpretation of comparison result."""
        interpretation = []
        
        # Significance
        if significant:
            interpretation.append(f"Statistically significant difference (p = {p_value:.4f})")
        else:
            interpretation.append(f"No statistically significant difference (p = {p_value:.4f})")
        
        # Effect size interpretation (Cohen's conventions)
        if effect_size is not None and not np.isnan(effect_size):
            abs_effect = abs(effect_size)
            if 'cohens_d' in test_name or 'ttest' in test_name:
                if abs_effect < 0.2:
                    size_interp = "negligible"
                elif abs_effect < 0.5:
                    size_interp = "small"
                elif abs_effect < 0.8:
                    size_interp = "medium"
                else:
                    size_interp = "large"
                interpretation.append(f"Effect size: {size_interp} (d = {effect_size:.3f})")
            else:
                interpretation.append(f"Effect size: {effect_size:.3f}")
        
        return ". ".join(interpretation)
    
    def generate_report(self, results: List[StatisticalTestResult]) -> str:
        """Generate a comprehensive statistical report."""
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        for i, result in enumerate(results, 1):
            report.append(f"Test {i}: {result.test_name}")
            report.append(f"  Statistic: {result.statistic:.4f}")
            report.append(f"  p-value: {result.p_value:.6f}")
            
            if result.effect_size is not None:
                report.append(f"  Effect size: {result.effect_size:.4f}")
            
            if result.confidence_interval is not None:
                ci_lower, ci_upper = result.confidence_interval
                report.append(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            if result.power is not None:
                report.append(f"  Statistical power: {result.power:.3f}")
            
            report.append(f"  Interpretation: {result.interpretation}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the statistical testing framework
    print("Testing Statistical Testing Framework...")
    
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(0, 1, 100)
    group2 = np.random.normal(0.5, 1.2, 120)
    group3 = np.random.normal(0.3, 0.8, 80)
    
    # Initialize tester
    tester = StatisticalTester(alpha=0.05)
    
    # Test normality
    norm_test = tester.test_normality(group1)
    print(f"✓ Normality test: {norm_test['interpretation']}")
    
    # Test two groups
    comparison = tester.compare_two_groups(group1, group2, test_type='auto')
    print(f"✓ Two-group comparison: {comparison.interpretation}")
    
    # Test multiple groups
    multi_comparison = tester.compare_multiple_groups([group1, group2, group3])
    print(f"✓ Multiple group comparison: {multi_comparison['overall_test']['interpretation']}")
    
    # Test bootstrap CI
    ci_lower, ci_upper, info = tester.bootstrap_confidence_interval(group1)
    print(f"✓ Bootstrap 95% CI for group1 mean: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test multiple comparison correction
    p_values = [0.01, 0.03, 0.08, 0.12, 0.001]
    corrected = tester.correct_multiple_comparisons(p_values, method='bonferroni')
    print(f"✓ Multiple comparison correction: {corrected['num_significant_corrected']} significant after correction")
    
    print("✓ Statistical testing framework test completed successfully!")