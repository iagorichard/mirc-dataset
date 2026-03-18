from utils.annotations import get_missing_video_annotation_data_info
from utils.files import check_data_integrity_extracted_frames

def main():
    missing_data = get_missing_video_annotation_data_info()
    check_data_integrity_extracted_frames(missing_data)

if __name__ == "__main__":
    main()