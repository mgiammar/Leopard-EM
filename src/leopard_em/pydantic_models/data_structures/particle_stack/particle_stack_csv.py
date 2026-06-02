"""CSV-backed ParticleStack implementation."""

from __future__ import annotations

import pandas as pd

from leopard_em.pydantic_models.formats import MATCH_TEMPLATE_DF_COLUMN_ORDER

from .base import _ParticleStackBase
from .particle_stack_hdf5 import ParticleStackHDF5
from .utils import _generate_particle_ids


class ParticleStackCSV(_ParticleStackBase):
    """Particle stack whose tabular data is loaded from a CSV file.

    Particle images are extracted from the micrograph paths referenced in the
    CSV at run time.  This is the original ``ParticleStack`` behavior.

    Attributes
    ----------
    df_path : str
        Path to the CSV file containing the particle data.
    """

    df_path: str

    def load_df(self) -> None:
        """Load and validate the particle DataFrame from ``df_path``.

        Raises
        ------
        ValueError
            If required columns are missing from the CSV.
        """
        tmp_df = pd.read_csv(self.df_path)

        missing_columns = [
            col for col in MATCH_TEMPLATE_DF_COLUMN_ORDER if col not in tmp_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing the following columns in DataFrame: {missing_columns}"
            )

        self._df = tmp_df

    def to_hdf5(
        self,
        hdf5_path: str,
        allow_file_overwrite: bool = False,
        include_image_stack: bool = False,
        include_local_stats: bool = False,
    ) -> ParticleStackHDF5:
        """Convert this CSV-backed stack to an HDF5-backed stack and write to disk.

        Parameters
        ----------
        hdf5_path : str
            Destination path for the HDF5 file.
        allow_file_overwrite : bool, optional
            Whether to overwrite an existing file, by default False.
        include_image_stack : bool, optional
            Write ``image_stack`` to the HDF5 file, by default False.
            Raises ``ValueError`` if the image stack has not been loaded.
        include_local_stats : bool, optional
            Write per-particle local stats to the HDF5 file, by default False.
            Raises ``ValueError`` if the local stats have not been set.

        Returns
        -------
        ParticleStackHDF5
            The new HDF5-backed stack instance pointing at ``hdf5_path``.
        """
        # Generate particle_id for each row and add to the copied DataFrame
        df = self._df.copy()
        particle_ids = _generate_particle_ids(df)
        df.insert(0, "particle_id", particle_ids)
        df = df.set_index("particle_id")
        df.index.name = "particle_id"

        hdf5_stack = ParticleStackHDF5(
            hdf5_path=hdf5_path,
            allow_file_overwrite=allow_file_overwrite,
            extracted_box_size=self.extracted_box_size,
            original_template_size=self.original_template_size,
            leopard_em_version=self.leopard_em_version,
            global_whitening_applied=self.global_whitening_applied,
            local_whitening_applied=self.local_whitening_applied,
            global_normalization_applied=self.global_normalization_applied,
            local_normalization_applied=self.local_normalization_applied,
            image_stack=self.image_stack if include_image_stack else None,
            local_stats_correlation_average=(
                self.local_stats_correlation_average if include_local_stats else None
            ),
            local_stats_correlation_variance=(
                self.local_stats_correlation_variance if include_local_stats else None
            ),
            skip_df_load=True,
        )
        hdf5_stack._df = df
        hdf5_stack.to_hdf5(
            include_image_stack=include_image_stack,
            include_local_stats=include_local_stats,
        )
        return hdf5_stack
