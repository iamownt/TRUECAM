import huggingface_hub
import os
from gigapath.preprocessing.data.slide_utils import find_level_for_target_mpp

# assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"

local_dir = os.path.join(os.path.expanduser("~"), ".cache/")
# huggingface_hub.hf_hub_download("prov-gigapath/prov-gigapath", filename="sample_data/PROV-000-000001.ndpi", local_dir=local_dir, force_download=True)

# slide_path = os.path.join(local_dir, "sample_data/PROV-000-000001.ndpi")
slide_path = "/home/user/sngp/tcga_slides/slides/TCGA-51-4079-01Z-00-DX1.111f7796-3797-4334-b543-918e374dbc22.svs"
# <_PropertyMap {'aperio.AppMag': '40', 'aperio.DSR ID': 'aperio01', 'aperio.Date': '06/20/11', 'aperio.DisplayColor': '0',
#                'aperio.Exposure Scale': '0.000001', 'aperio.Exposure Time': '109', 'aperio.Filename': '26726', 'aperio.Focus Offset': '0.000000',
#                'aperio.ICC Profile': 'ScanScope v1', 'aperio.ImageID': '26726', 'aperio.Left': '23.843460', 'aperio.LineAreaXOffset': '0.000000',
#                'aperio.LineAreaYOffset': '0.000000', 'aperio.LineCameraSkew': '-0.000389', 'aperio.MPP': '0.2520', 'aperio.OriginalHeight': '63225',
#                'aperio.OriginalWidth': '107000', 'aperio.Parmset': 'EPC', 'aperio.ScanScope ID': 'SS1302', 'aperio.StripeWidth': '1000',
#                'aperio.Time': '13:06:27', 'aperio.Time Zone': 'GMT-07:00', 'aperio.Top': '21.591766', 'aperio.User': 'bf4e4f93-d36f-4dd6-9328-5cbbf1164855',
#                'openslide.associated.thumbnail.height': '629', 'openslide.associated.thumbnail.width': '1024',
#                'openslide.comment': 'Aperio Image Library v10.2.41\r\n107000x63225 [0,100 102639x63125] (256x256) J2K/YUV16 Q=70|AppMag = 40|StripeWidth = 1000|ScanScope ID = SS1302|Filename = 26726|Date = 06/20/11|Time = 13:06:27|Time Zone = GMT-07:00|User = bf4e4f93-d36f-4dd6-9328-5cbbf1164855|Parmset = EPC|MPP = 0.2520|Left = 23.843460|Top = 21.591766|LineCameraSkew = -0.000389|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|Focus Offset = 0.000000|DSR ID = aperio01|ImageID = 26726|Exposure Time = 109|Exposure Scale = 0.000001|DisplayColor = 0|OriginalWidth = 107000|OriginalHeight = 63225|ICC Profile = ScanScope v1', 'openslide.icc-size': '141992', 'openslide.level-count': '4', 'openslide.level[0].downsample': '1', 'openslide.level[0].height': '63125', 'openslide.level[0].tile-height': '256', 'openslide.level[0].tile-width': '256', 'openslide.level[0].width': '102639', 'openslide.level[1].downsample': '4.0000901426904631', 'openslide.level[1].height': '15781', 'openslide.level[1].tile-height': '256', 'openslide.level[1].tile-width': '256', 'openslide.level[1].width': '25659', 'openslide.level[2].downsample': '16.001803030680271', 'openslide.level[2].height': '3945', 'openslide.level[2].tile-height': '256', 'openslide.level[2].tile-width': '256', 'openslide.level[2].width': '6414', 'openslide.level[3].downsample': '32.007663177848158', 'openslide.level[3].height': '1972', 'openslide.level[3].tile-height': '256', 'openslide.level[3].tile-width': '256', 'openslide.level[3].width': '3207', 'openslide.mpp-x': '0.252', 'openslide.mpp-y': '0.252', 'openslide.objective-power': '40', 'openslide.quickhash-1': '9ee296701013507f77dc0a4946ad3354a75784b2639672773d81eef9968ab235', 'openslide.vendor': 'aperio', 'tiff.ImageDescription': 'Aperio Image Library v10.2.41\r\n107000x63225 [0,100 102639x63125] (256x256) J2K/YUV16 Q=70|AppMag = 40|StripeWidth = 1000|ScanScope ID = SS1302|Filename = 26726|Date = 06/20/11|Time = 13:06:27|Time Zone = GMT-07:00|User = bf4e4f93-d36f-4dd6-9328-5cbbf1164855|Parmset = EPC|MPP = 0.2520|Left = 23.843460|Top = 21.591766|LineCameraSkew = -0.000389|LineAreaXOffset = 0.000000|LineAreaYOffset = 0.000000|Focus Offset = 0.000000|DSR ID = aperio01|ImageID = 26726|Exposure Time = 109|Exposure Scale = 0.000001|DisplayColor = 0|OriginalWidth = 107000|OriginalHeight = 63225|ICC Profile = ScanScope v1', 'tiff.ResolutionUnit': 'inch'}>


print("NOTE: Prov-GigaPath is trained with 0.5 mpp preprocessed slides")
target_mpp = 0.5
level = find_level_for_target_mpp(slide_path, target_mpp)
if level is not None:
    print(f"Found level: {level}")
else:
    print("No suitable level found.")


