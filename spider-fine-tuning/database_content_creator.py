from typing import Iterable, List, Optional

import sqlalchemy
from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.schema import CreateTable

import warnings
warnings.filterwarnings("ignore")


def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )

class FineTuningDatabaseContentCreator:
    def __init__(
        self,
        engine: Engine,
        schema: str | None = None,
        metadata: MetaData | None = None,
        ignore_tables: List[str] | None = None,
        include_tables: List[str] | None = None,
        sample_rows_in_table_info: int = 3,
        low_cardinality_threshold: int = 10,
        indexes_in_table_info: bool = False,
        custom_table_info: dict | None = None,
        view_support: bool = False,
        max_string_length: int = 300,
    ):
        self._engine = engine
        self._schema = schema

        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)

        # including view support by adding the views as well as tables to the all
        # tables list if view_support is True
        self._all_tables = set(
            self._inspector.get_table_names(schema=schema)
            + (self._inspector.get_view_names(schema=schema) if view_support else [])
        )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
            
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )

        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")
        
        if not isinstance(low_cardinality_threshold, int):
            raise TypeError("low_cardinality_threshold must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._low_cardinality_threshold = low_cardinality_threshold
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        self._max_string_length = max_string_length

        self._metadata = metadata or MetaData()
        # including view support if view_support = true
        self._metadata.reflect(
            views=view_support,
            bind=self._engine,
            only=list(self._usable_tables),
            schema=self._schema,
        )

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)
    
    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """
        Get information about specified tables.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                try:
                    table_info += f"\n{self._get_sample_rows(table)}\n"
                except Exception:
                    pass
            tables.append(table_info)
        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str
    
    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    def _get_sample_rows(self, table: Table) -> str:

        limiting_factor = 200
        # build the select command
        command = select(table).limit(limiting_factor)

        try:
            with self._engine.connect() as connection:
                response = ""
                sample_rows_result = connection.execute(command)
                sample_rows = sample_rows_result.fetchall()

                 # Create sections for high and low cardinality columns
                high_cardinality_section = f"/*\nColumns in {table.name} and {str(self._sample_rows_in_table_info)} examples in each column for high cardinality columns :"  # noqa: E501
                low_cardinality_section = f"/*\nColumns in {table.name} and all categories for low cardinality columns :"  # noqa: E501

                low_columns = ""
                high_columns = ""
                
                for column, index in zip(table.columns,range(len(table.columns))):
                    column_name = column.name
                    values = [str(row[index]) for row in sample_rows]

                    # Determine if the column is high or low cardinality based on the threshold  # noqa: E501
                    unique_values = set(values)
                    if len(unique_values) > self._low_cardinality_threshold:
                        high_columns += f"\n{column_name} : {', '.join(list(unique_values)[:self._sample_rows_in_table_info])}"  # noqa: E501
                    else:
                        low_columns += f"\n{column_name} : {', '.join(unique_values)}"  # noqa: E501

                if high_columns:
                    high_cardinality_section += high_columns + "\n*/\n"
                    response += high_cardinality_section

                if low_columns:
                    low_cardinality_section += low_columns + "\n*/"
                    response += low_cardinality_section

        except ProgrammingError:
                response = ""

        return response
