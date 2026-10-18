import unittest

import pandas as pd

from utils import clean_sales_data, engineer_features


class CleanSalesDataTests(unittest.TestCase):
    def test_cleaning_options_fill_missing_and_remove_exact_duplicates(self):
        source = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "sales": [10, 10, None],
                "family": ["A", "A", "A"],
            }
        )

        cleaned = clean_sales_data(source, fill_missing_sales=True)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["sales"].tolist(), [10, 0])
        self.assertTrue(source["sales"].isna().iloc[2])

    def test_cleaning_can_preserve_duplicates_and_drop_missing_sales(self):
        source = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "sales": [10, 10, None],
            }
        )

        cleaned = clean_sales_data(
            source, fill_missing_sales=False, remove_duplicates=False
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["sales"].tolist(), [10, 10])

    def test_conflicting_duplicate_ids_raise_instead_of_discarding_data(self):
        source = pd.DataFrame(
            {
                "id": [1, 1],
                "date": ["2024-01-01", "2024-01-01"],
                "sales": [10, 12],
            }
        )

        with self.assertRaisesRegex(ValueError, "trùng id"):
            clean_sales_data(source)

    def test_winsorize_caps_sales_at_one_and_ninety_ninth_percentiles(self):
        source = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=101),
                "sales": [*range(100), 10_000],
            }
        )
        lower, upper = source["sales"].quantile([0.01, 0.99])

        cleaned = clean_sales_data(source, cap_outliers=True)

        self.assertEqual(cleaned["sales"].min(), lower)
        self.assertEqual(cleaned["sales"].max(), upper)
        self.assertEqual(source["sales"].max(), 10_000)

    def test_empty_cleaned_data_raises_instead_of_succeeding(self):
        source = pd.DataFrame({"date": ["not-a-date"], "sales": [None]})

        with self.assertRaisesRegex(ValueError, "Không còn dòng hợp lệ"):
            clean_sales_data(source, fill_missing_sales=False)

    def test_feature_engineering_runs_on_cleaned_rows(self):
        source = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "sales": [10, 12],
                "family": ["A", "A"],
            }
        )

        processed = engineer_features(clean_sales_data(source))

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed["lag_1"].iloc[1], 10)


if __name__ == "__main__":
    unittest.main()
