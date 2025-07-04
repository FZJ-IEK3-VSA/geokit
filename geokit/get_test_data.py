import pathlib
from typing import Literal

import pooch

all_file_name_dict = {
    "aachenShapefile.dbf": "sha256:0f1262b987e88fe3eef267b828d4b6712a7ba71fe22a995c2a67d4a8a3200292",
    "aachenShapefile.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachenShapefile.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "aachenShapefile.shp": "sha256:fdf082c0c6adb0c00332259c43455598562c1a534feb105e53ee69b8646984de",
    "aachenShapefile.shx": "sha256:5de9710f210b104fc3573a04e480d8b1c49784370c117d32f4a89a88c8a6f0c6",
    "aachen_buildings.dbf": "sha256:9ab2584c73c9497ca93bf9a8d64355f3fcb7e8afaf3653561a302ad69243e53b",
    "aachen_buildings.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_buildings.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "aachen_buildings.shp": "sha256:e60940a1f91e916b7b9ab0b789be4e771d7c2d3063f19b4b17c2d14a72682dcf",
    "aachen_buildings.shx": "sha256:e335225ae6a2c65c823b0d97cbec1eb97065a5b49b7456c176f8c910c5cadee8",
    "aachen_eligibility.tif": "sha256:2cdddce2a97c8b314d74c3e060fef07a15e1cfe995804bc4202aad77bc268748",
    "aachen_points.dbf": "sha256:85cf4e0c26e043342aee9cd5ddc0d11c1355d0a8e0cb3dafd6274e62bfca59c0",
    "aachen_points.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_points.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "aachen_points.shp": "sha256:3e07f2a1664b88482e576dad04a94ec25c1e16985f86a27b526970159aef0663",
    "aachen_points.shx": "sha256:7057b4c38e18afe60a0d1a4b06f715af7611a908ba6d89d9a9adf5a0371a5a22",
    "aachen_rails.dbf": "sha256:f8ea646fa52d507d218d3659265b35bac3a86154c0f53ba89ee1f37a011302d8",
    "aachen_rails.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "aachen_rails.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "aachen_rails.shp": "sha256:fa4495ca0fb31f37db30b4e64a4d00ac7af68e9df17c72d2ba9027a4ee95723b",
    "aachen_rails.shx": "sha256:1f2984541fdd9d3088f2c2c05211d6509b703707f18eb643fe1bb7cae341318c",
    "aachen_zones.cpg": "sha256:3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "aachen_zones.dbf": "sha256:bf99f3e355b064ba1515ce28168631467d60e1994fbb0a1c94f818662a255efc",
    "aachen_zones.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "aachen_zones.qpj": "sha256:ade308b71f1982004ec42becd8be9c2c634965a4a0e99a980a4ab707d0919649",
    "aachen_zones.shp": "sha256:beb492bcc7e113abf00ca6d3c92c639e76c79346d9161e84251d6c4b794318fa",
    "aachen_zones.shx": "sha256:97aa30327616e7d7d89c2a88b4ac2665da4a734906fe72e27ad08531f337e30e",
    "boxes.dbf": "sha256:594bb714232471af4d4ead34b0887a86f713478d3468572a2bae9a4fa3a51bae",
    "boxes.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "boxes.shp": "sha256:6c162047ba0f9d2a9c59c3fa9aad43931df52364d49856d98d267f3f4a580c44",
    "boxes.shx": "sha256:5bdd1a01fe9eb5876ab6a9db32c9dbba18611b909bfbcd48a5397983350430cf",
    "CDDA_aachenClipped.dbf": "sha256:aaef0d03954f2fdc1b1f0570b6ba9e7fb1929624c5200b7d5db5ef40f5562e84",
    "CDDA_aachenClipped.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "CDDA_aachenClipped.qpj": "sha256:ade308b71f1982004ec42becd8be9c2c634965a4a0e99a980a4ab707d0919649",
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
    "Europe_with_H2MobilityData_GermanyClip.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
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
    "LuxLines.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "LuxLines.shp": "sha256:245551e6c8b8571ae075576b64580293cf8bdc6e1100f72141841589ed6b1f98",
    "LuxLines.shx": "sha256:929b2c9c67a4d1cf50c5a07949ebe079f44fe8f15fcf6a77f10682acfbfc2d47",
    "LuxShape.dbf": "sha256:56884e2cf9f2673c60c1b07d5b10cd915db0ad305b2ab6bbe661d3ee57a84ed1",
    "LuxShape.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "LuxShape.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "LuxShape.shp": "sha256:1b86cae6fbf40854fb1c80d1365cdcb431d95c62c8dbb1c3c8a3d439bc33c29c",
    "LuxShape.shx": "sha256:5e02420281139c2a6aacea8836010f00f6be30df329be7975ee1916155964d40",
    "multiFeature.cpg": "sha256:3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
    "multiFeature.dbf": "sha256:f55f8c7a9af335c313f1bebd5c403c882329b23b8790807712ef2769153c5091",
    "multiFeature.prj": "sha256:98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb",
    "multiFeature.qpj": "sha256:6eb7e7bea6a396821b1c484f05584d4afa8aff85898e79faefc63029fa9f2962",
    "multiFeature.shp": "sha256:19aeea0546f2bb1f5c5a2d1183168524cda03ab945e1625b5ccef0e5374a1d3b",
    "multiFeature.shx": "sha256:82f36730335e51189a5fbd88d145e98924958be92a60182cbf6fcecaf4a70aa9",
    "Natura2000_aachenClipped.dbf": "sha256:d36e38b42d8ee567baea20dd0cab76119c566422967648f5a216089181d4fdc2",
    "Natura2000_aachenClipped.prj": "sha256:1fb8a9c4eb5c8c90031c66d4192f0f67bb09b47d29a9e055df3f0be188864db8",
    "Natura2000_aachenClipped.qpj": "sha256:ade308b71f1982004ec42becd8be9c2c634965a4a0e99a980a4ab707d0919649",
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
}


def get_test_data(
    file_name: str,
    data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath(
        "data"
    ),
    no_download: bool = True,
) -> str:

    if file_name not in all_file_name_dict:
        raise Exception(
            "The requested file,"
            + str(file_name)
            + " , is not included in the test and example data dictionary. Perhaps it's a typo? The following files can be retrieved from the test and example data dictionary: "
            + str(list(all_file_name_dict.keys()))
        )

    if no_download is False:
        odie = pooch.create(
            # Use the default cache folder for the operating system
            path=data_cache_folder,
            base_url="https://zenodo.org/records/11032664/preview/",
            # The registry specifies the files that can be fetched
            registry=all_file_name_dict,
        )
        return_path = odie.fetch(fname=file_name)
    else:
        return_path = data_cache_folder.joinpath(file_name)
        if not return_path.is_file():
            raise Exception("There is no file at: " + str(return_path))
        file_hash = pooch.file_hash(return_path, alg="sha256")
        file_hash_with_algorithm_prefix = "sha256:" + file_hash
        file_hash_stored = all_file_name_dict[file_name]
        assert file_hash_stored == file_hash_with_algorithm_prefix, (
            "There is a hash mismatch between the actual file and the stored hash: "
            + str(return_path)
            + ". The stored hash is: "
            + file_hash_stored
            + " and the calculated hash is: "
            + file_hash_with_algorithm_prefix
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


def get_test_shape_file(
    file_name_without_extension: str,
    extension: Literal[
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
    ],
    data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath(
        "data"
    ),
    no_download: bool = True,
) -> str:
    file_name = file_name_without_extension + extension
    return_path = get_test_data(
        file_name=file_name,
        data_cache_folder=data_cache_folder,
        no_download=no_download,
    )

    for additional_file_type in list_of_all_shape_file_extensions:
        additional_file_name = file_name_without_extension + additional_file_type
        if additional_file_name in all_file_name_dict:
            get_test_data(
                file_name=additional_file_name,
                data_cache_folder=data_cache_folder,
                no_download=no_download,
            )
    return return_path
