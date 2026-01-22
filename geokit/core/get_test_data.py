import hashlib
import os
import pathlib
from collections import OrderedDict as _OrderedDict
from typing import Literal


import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

all_file_name_dict = {
    "aachenShapefile.dbf": "sha256:0f1262b987e88fe3eef267b828d4b6712a7ba71fe22a995c2a67d4a8a3200292",
    "aachenShapefile.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachenShapefile.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "aachenShapefile.shp": "sha256:fdf082c0c6adb0c00332259c43455598562c1a534feb105e53ee69b8646984de",
    "aachenShapefile.shx": "sha256:5de9710f210b104fc3573a04e480d8b1c49784370c117d32f4a89a88c8a6f0c6",
    "aachen_buildings.dbf": "sha256:9ab2584c73c9497ca93bf9a8d64355f3fcb7e8afaf3653561a302ad69243e53b",
    "aachen_buildings.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_buildings.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "aachen_buildings.shp": "sha256:e60940a1f91e916b7b9ab0b789be4e771d7c2d3063f19b4b17c2d14a72682dcf",
    "aachen_buildings.shx": "sha256:e335225ae6a2c65c823b0d97cbec1eb97065a5b49b7456c176f8c910c5cadee8",
    "aachen_eligibility.tif": "sha256:2cdddce2a97c8b314d74c3e060fef07a15e1cfe995804bc4202aad77bc268748",
    "aachen_points.dbf": "sha256:85cf4e0c26e043342aee9cd5ddc0d11c1355d0a8e0cb3dafd6274e62bfca59c0",
    "aachen_points.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_points.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "aachen_points.shp": "sha256:3e07f2a1664b88482e576dad04a94ec25c1e16985f86a27b526970159aef0663",
    "aachen_points.shx": "sha256:7057b4c38e18afe60a0d1a4b06f715af7611a908ba6d89d9a9adf5a0371a5a22",
    "aachen_rails.dbf": "sha256:f8ea646fa52d507d218d3659265b35bac3a86154c0f53ba89ee1f37a011302d8",
    "aachen_rails.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_rails.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "aachen_rails.shp": "sha256:fa4495ca0fb31f37db30b4e64a4d00ac7af68e9df17c72d2ba9027a4ee95723b",
    "aachen_rails.shx": "sha256:1f2984541fdd9d3088f2c2c05211d6509b703707f18eb643fe1bb7cae341318c",
    "aachen_zones.cpg": "sha256:3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "aachen_zones.dbf": "sha256:bf99f3e355b064ba1515ce28168631467d60e1994fbb0a1c94f818662a255efc",
    "aachen_zones.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "aachen_zones.qpj": "sha256:ac0f80bae3da64638fa7c3567f219f166d22171efe582d7dbe16814e795b6c32",
    "aachen_zones.shp": "sha256:beb492bcc7e113abf00ca6d3c92c639e76c79346d9161e84251d6c4b794318fa",
    "aachen_zones.shx": "sha256:97aa30327616e7d7d89c2a88b4ac2665da4a734906fe72e27ad08531f337e30e",
    "boxes.dbf": "sha256:594bb714232471af4d4ead34b0887a86f713478d3468572a2bae9a4fa3a51bae",
    "boxes.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "boxes.shp": "sha256:6c162047ba0f9d2a9c59c3fa9aad43931df52364d49856d98d267f3f4a580c44",
    "boxes.shx": "sha256:5bdd1a01fe9eb5876ab6a9db32c9dbba18611b909bfbcd48a5397983350430cf",
    "CDDA_aachenClipped.dbf": "sha256:aaef0d03954f2fdc1b1f0570b6ba9e7fb1929624c5200b7d5db5ef40f5562e84",
    "CDDA_aachenClipped.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "CDDA_aachenClipped.qpj": "sha256:ac0f80bae3da64638fa7c3567f219f166d22171efe582d7dbe16814e795b6c32",
    "CDDA_aachenClipped.shp": "sha256:0448409e07c8cecd0874d48ae5b6dfcdc63638fb0ad82db53a83690b3ecaec9f",
    "CDDA_aachenClipped.shx": "sha256:5cf6749ee0e4647b7b543cd6b60be8e5bc1edf90be3247d6b2c189e3cc88d1f4",
    "clc-aachen_clipped-unflipped.tif": "sha256:a0a69c37c4bc09af44f701aead44e8468b9f1c6e7a0bc4874b071ffdfb691473",
    "clc-aachen_clipped.tif": "sha256:16c09e4a0e07cd9c1fa1674af6b5a652b86010e010174c99177f0d51438657ae",
    "clippedCLC.tif": "sha256:452bb2c889a809370fdc2753a148aa02d2d365695cd58e201c359c17bbdff913",
    "divided_raster_1.tif": "sha256:3a5f99a8e202911fcecad337e3e695ce694221c66b0c073b69d14788656afe35",
    "divided_raster_2.tif": "sha256:e62e1b9ded724757c0626934fbf7f467c8386a90fcb803bced3731801b583c54",
    "divided_raster_3.tif": "sha256:3742c40d6a50bd1e04b2d66f8a598d3246fe09152e1497a9c578de21c2090ff0",
    "elevation.tif": "sha256:851f06b5b2a58d57a6f8a5e449e7d4879b3499f728c784794b8edb95eee15aa9",
    "elevation_singleHill.tif": "sha256:a6d2fa784babd2381a7f3f42fbd159f3c76ac3f998b987fb119142a08a9e7283",
    "Europe_with_H2MobilityData_GermanyClip.dbf": "sha256:bc226aff7b50c5401e8295e7ad4af2397a5ad17d3d118270df6406787e1a0c0f",
    "Europe_with_H2MobilityData_GermanyClip.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "Europe_with_H2MobilityData_GermanyClip.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "Europe_with_H2MobilityData_GermanyClip.shp": "sha256:54ac326203d6eea777b63198b81a1ec8c34c8a36c98dfa6c826ebfb944c04881",
    "Europe_with_H2MobilityData_GermanyClip.shx": "sha256:dae162c254e93cc94ad0879ca56a41b72c1af9524bbd2d103b19d8bd09a751f0",
    "FJI.dbf": "sha256:adf0ec8ecc720c1a2b693c88c5b11c126dfb20112b64f458954ed3ebd407c5c8",
    "FJI.prj": "sha256:a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c4",
    "FJI.shp": "sha256:4c50ae5a90593c5036660afcd98e0d7a0b6cc06f65628592443e286fd35f4ce9",
    "FJI.shx": "sha256:7e38e17879886f96721810fb0e8980ca2a363791dc2f579853fc7930079e7f8b",
    "gadm36_DEU_1.cpg": "sha256:3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "gadm36_DEU_1.dbf": "sha256:bc039c616cbb7759b9090bd300956b6e06d0dbe48bff20e42e702bc80946e464",
    "gadm36_DEU_1.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "gadm36_DEU_1.shp": "sha256:0e2ab0710f9cbddbddc57b8f4b32ff71da06445e0f8acf47496c6043c472502f",
    "gadm36_DEU_1.shx": "sha256:cb5f2eb951540ee342033ffb30a4ad6e5af0c3d1dfee3e3cf0cd43b975baa969",
    "gsa-ghi-like.tif": "sha256:af432e2eddd8fcfee34c60dfdc6ba2fa61f34db1d80e4489443acb0d556b6e80",
    "LuxLines.dbf": "sha256:1e811272c479f24566105b1bb7055049f111318899a676c559b295cbcf3e86fa",
    "LuxLines.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "LuxLines.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "LuxLines.shp": "sha256:245551e6c8b8571ae075576b64580293cf8bdc6e1100f72141841589ed6b1f98",
    "LuxLines.shx": "sha256:929b2c9c67a4d1cf50c5a07949ebe079f44fe8f15fcf6a77f10682acfbfc2d47",
    "LuxShape.dbf": "sha256:56884e2cf9f2673c60c1b07d5b10cd915db0ad305b2ab6bbe661d3ee57a84ed1",
    "LuxShape.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "LuxShape.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "LuxShape.shp": "sha256:1b86cae6fbf40854fb1c80d1365cdcb431d95c62c8dbb1c3c8a3d439bc33c29c",
    "LuxShape.shx": "sha256:5e02420281139c2a6aacea8836010f00f6be30df329be7975ee1916155964d40",
    "multiFeature.cpg": "sha256:3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "multiFeature.dbf": "sha256:f55f8c7a9af335c313f1bebd5c403c882329b23b8790807712ef2769153c5091",
    "multiFeature.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "multiFeature.qpj": "sha256:1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657",
    "multiFeature.shp": "sha256:19aeea0546f2bb1f5c5a2d1183168524cda03ab945e1625b5ccef0e5374a1d3b",
    "multiFeature.shx": "sha256:82f36730335e51189a5fbd88d145e98924958be92a60182cbf6fcecaf4a70aa9",
    "Natura2000_aachenClipped.dbf": "sha256:d36e38b42d8ee567baea20dd0cab76119c566422967648f5a216089181d4fdc2",
    "Natura2000_aachenClipped.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "Natura2000_aachenClipped.qpj": "sha256:ac0f80bae3da64638fa7c3567f219f166d22171efe582d7dbe16814e795b6c32",
    "Natura2000_aachenClipped.shp": "sha256:03b6c467e02ec69b6e58f3e1a54b17cad195a9f6ef1cf24f4c71f1a753eadf38",
    "Natura2000_aachenClipped.shx": "sha256:3fad9841ad7af23c58775bf149719c0053161bead15f9b0b3dcc8b3a734331e2",
    "raster_gdal_244.tif": "sha256:14cc7ec162b42052bd201e731a3b75a3c9a18c21c3f24a8167a9abac88528c13",
    "surroundingRaster.tif": "sha256:8fd025aab0f5c97388265ab6d70ed0c3d4a5898bb669b42e3b9c58f7a172adb1",
    "turbinePlacements.dbf": "sha256:ba82065e0a78681ffd6b8d10e85b9276a9e7aad84b18ca1cc916ea1327785234",
    "turbinePlacements.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "turbinePlacements.shp": "sha256:4e2fbd2ef70575ebf5c812197f5dec0e472c11ad9124985fd3e11eeff81b094c",
    "turbinePlacements.shx": "sha256:1dbd48d0a65724aa5fb94741a595d0691646750f707fb040589aa38a29ac7406",
    "urban_land_cover_aachenClipped.tif": "sha256:a108016886bdb5b3280741c3040a49fd0422ef3a61e46601bef82f10236f5af1",
    "osm_roads_minor.9.264.171.tif": "sha256:6fe1632758d39a5300dd189238d49bbf933d10c0b2c9b7d3b6b68b264d732be8",
    "osm_roads_minor.9.264.172.tif": "sha256:d7f54eabf297458f8f43cd1aec745c4710d7a1e2c26fed0ef3345625e3f30fa0",
    "osm_roads_minor.9.265.171.tif": "sha256:68a77d446fa5fb27b05c068d24214bd2a18135e1231b686746387b8f2aa88681",
    "osm_roads_minor.9.265.172.tif": "sha256:2a5029eee67d74a1f1e2c0d27b786a757e8687375ef4190ca2daab052aa8302e",
    "test_raster_3x3.tif": "sha256:49e4f41d636618ef38d1a6cb6f684af2d90df4735aeaea88340f3a7fb51b1a84",
}


ZENODO_PREVIEW_BASE_URL = "https://zenodo.org/records/11032664/preview/"
_HASH_BUFFER_SIZE = 1024 * 1024  # 1MB chunks for hashing


def _compute_file_hash(file_path: pathlib.Path, alg: str = "sha256") -> str:
    hasher = hashlib.new(alg)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_BUFFER_SIZE), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_file_hash(file_path: pathlib.Path, stored_hash: str) -> None:
    if ":" in stored_hash:
        alg, expected_hash = stored_hash.split(":", 1)
    else:
        alg, expected_hash = "sha256", stored_hash
    calculated_hash = _compute_file_hash(file_path, alg=alg)
    assert expected_hash == calculated_hash, (
        "There is a hash mismatch between the actual file and the stored hash: "
        + str(file_path)
        + ". The stored hash is: "
        + stored_hash
        + " and the calculated hash is: "
        + alg
        + ":"
        + calculated_hash
    )


def _get_test_data(
    file_name: str,
    data_cache_folder: pathlib.Path,
) -> str:
    return_path = data_cache_folder.joinpath(file_name)
    if not return_path.is_file():
        raise Exception("There is no file at: " + str(return_path))

    _verify_file_hash(
        file_path=return_path,
        stored_hash=all_file_name_dict[file_name],
    )
    return_path_str = str(return_path)
    return return_path_str


list_of_all_shape_file_extensions = [
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
    ".sbn",
    ".sbx",
    ".ain",
    ".aih",
    ".ixs",
    ".mxs",
    ".atx",
    ".shp.xml",
    ".cpg",
    ".qix",
]


def get_test_data(
    file_name: str,
    data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("data"),
) -> str:
    if file_name not in all_file_name_dict:
        raise Exception(
            "The requested file,"
            + str(file_name)
            + " , is not included in the test and example data dictionary. Perhaps it's a typo? The following files can be retrieved from the test and example data dictionary: \n\n"
            + str(list(all_file_name_dict.keys()))
        )
    file_extension = pathlib.Path(file_name).suffix
    file_name = pathlib.Path(file_name).name

    return_path = _get_test_data(
        file_name=file_name,
        data_cache_folder=data_cache_folder,
    )
    return_path_str = str(return_path)
    if file_extension in list_of_all_shape_file_extensions:
        for additional_shape_file_extension in list_of_all_shape_file_extensions:
            additional_file_name = str(file_name + str(additional_shape_file_extension))
            if additional_file_name in all_file_name_dict:
                _get_test_data(
                    file_name=additional_file_name,
                    data_cache_folder=data_cache_folder,
                )

    return return_path_str


def get_all_shape_files(
    data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("data"),
):
    for current_file in all_file_name_dict.keys():
        get_test_data(
            file_name=current_file,
            data_cache_folder=data_cache_folder,
        )
    path_to_all_shape_files = data_cache_folder.joinpath("*.shp")
    return path_to_all_shape_files


def create_hash_dict(list_of_file_paths: list[pathlib.Path], alg: str = "sha256") -> dict[str, str]:
    output_dict = {}
    for current_file_path in list_of_file_paths:
        file_hash = _compute_file_hash(current_file_path, alg=alg)
        output_dict[current_file_path.name] = alg + ":" + file_hash
    return output_dict


def get_all_test_data_dict() -> _OrderedDict[str, str]:
    _test_data_ = _OrderedDict()
    for current_file_name in all_file_name_dict.keys():
        _test_data_[current_file_name] = get_test_data(file_name=current_file_name)
    return _test_data_


class ZenodoDataDownloader:
    def __init__(
        self,
        data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath("data"),
    ):
        self.data_cache_folder = data_cache_folder

    def get_zenodo_header(self) -> dict[str, str]:
        user_agent_string = os.getenv("ZENODO_USER_AGENT_STRING")
        api_key = os.getenv("ZENODO_API_KEY")

        headers = {}
        if isinstance(user_agent_string, str):
            headers["User-Agent"] = user_agent_string
        if isinstance(api_key, str):
            headers["Authorization"] = "Bearer " + api_key
        return headers

    def download_file(
        self,
        url: str,
        filename: str | None = None,
        headers: dict | None = None,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Download a single file to the cache folder.

        If ``filename`` is omitted, it is derived from the URL path. Set
        ``overwrite`` to re-download an existing file.
        """
        parsed = urlparse(url)
        derived_name = pathlib.Path(parsed.path).name or "download"
        target_name = filename if isinstance(filename, str) else derived_name
        target_path = self.data_cache_folder.joinpath(target_name)

        if target_path.exists() and not overwrite:
            print(f"File already exists at {target_path}, skipping download.")
            return target_path

        if headers is None:
            headers_internal = {}
        else:
            headers_internal = headers

        self.data_cache_folder.mkdir(parents=True, exist_ok=True)
        with requests.get(url, timeout=300, allow_redirects=True, headers=headers_internal, stream=True) as response:
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"Downloaded to {target_path}")
        return target_path

    def extract_zip_archive(self, path_to_archive: pathlib.Path | str, extract_folder: str):
        if isinstance(path_to_archive, str):
            path_to_archive = pathlib.Path(path_to_archive)
        if not path_to_archive.is_file():
            raise Exception(f"Archive missing: {path_to_archive}")
        if path_to_archive.suffix != ".zip":
            raise Exception(f"Not a zip archive: {path_to_archive}. Only zip archives are supported.")

        extract_folder_path = self.data_cache_folder.joinpath(extract_folder)
        extract_folder_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path_to_archive, "r") as zf:
            members_to_extract = []
            for m in zf.infolist():
                target_path = extract_folder_path / m.filename
                if not target_path.is_file() or target_path.stat().st_size != m.file_size:
                    members_to_extract.append(m)

            if not members_to_extract:
                print(f"Already extracted to {extract_folder_path}")
                return extract_folder_path

            for m in members_to_extract:
                zf.extract(m, extract_folder_path)
            print(f"Extracted to {extract_folder_path}")

        return extract_folder_path

    def download_and_extract_parallel(
        self,
        download_list: list[tuple[str, str | None, str | pathlib.Path | None, dict | None]],
        max_workers: int = 4,
    ) -> list[pathlib.Path]:
        """Parallelize multiple downloads with optional extraction per job.


        Each download_list entry: (url, filename_or_none, extract_dir_or_none, header).
        If extract_dir is provided, the downloaded file is extracted there (zip only).
        Returns paths to the downloaded files or extraction folders in input order.
        """

        def _worker(idx: int, job: tuple[str, str | None, str | None, dict | None]) -> tuple[int, pathlib.Path]:
            url, filename, extract_dir, headers = job
            downloaded_path = self.download_file(url=url, filename=filename, headers=headers)
            if extract_dir is None:
                return idx, downloaded_path
            extracted_path = self.extract_zip_archive(
                path_to_archive=downloaded_path,
                extract_folder=extract_dir,
            )
            return idx, extracted_path

        indexed_jobs = list(enumerate(download_list))
        results: list[pathlib.Path | None] = [None for _ in download_list]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, idx, job): idx for idx, job in indexed_jobs}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    out_idx, path = future.result()
                except Exception as exc:  # bubble up with context
                    raise Exception(f"Batch job failed for index {idx}: {exc}") from exc
                results[out_idx] = path

        return [path for path in results if path is not None]
