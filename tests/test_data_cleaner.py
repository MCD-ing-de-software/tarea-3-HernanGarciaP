import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name", "age"])

        # El resultado no debe tener valores faltantes en "name" ni en "age"
        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)

        # El resultado debe tener menos filas que el DataFrame original
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        # "does_not_exist" no es una columna del DataFrame → debe lanzar KeyError
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco al inicio y final, sin modificar el DataFrame original ni otras columnas.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        result = cleaner.trim_strings(df, ["name"])

        # El DataFrame original NO debe haber sido modificado (inmutabilidad)
        self.assertEqual(df["name"].iloc[0], " Alice ")
        self.assertEqual(df["name"].iloc[3], " Carol  ")

        # En el resultado, los valores de "name" no deben tener espacios al inicio/final
        self.assertEqual(result["name"].iloc[0], "Alice")
        self.assertEqual(result["name"].iloc[3], "Carol")

        # La columna "city" (no especificada) debe permanecer sin cambios
        pdt.assert_series_equal(result["city"], df["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna que no es de tipo string.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        # "age" es una columna numérica → debe lanzar TypeError
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que el método remove_outliers_iqr elimina correctamente los
        valores extremos (outliers) de una columna numérica usando el método IQR.
        """
        # Usamos un DataFrame sin NaN en "age" para que el IQR se calcule correctamente
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol", "Dave"],
            "age": [25, 30, 35, 120],  # 120 es un outlier claro
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        })
        cleaner = DataCleaner()

        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)

        # El valor extremo 120 debe haber sido eliminado
        self.assertNotIn(120, result["age"].values)

        # Al menos uno de los valores normales debe permanecer
        self.assertIn(25, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        # "salary" no existe en el DataFrame → debe lanzar KeyError
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un TypeError cuando
        se llama con una columna que no es de tipo numérico.
        """
        df = make_sample_df()
        cleaner = DataCleaner()

        # "city" es una columna de texto → debe lanzar TypeError
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")


if __name__ == "__main__":
    unittest.main()
