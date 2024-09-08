




def copy_files_to_directory(file_paths, dest_dir):
    """
    Copy a list of files to a destination directory, creating the directory if it doesn't exist.

    Parameters:
    -----------
    file_paths : list
        List of file paths to be copied.
    dest_dir : str or Path
        Destination directory to copy the files to.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    print(f"Copying {len(file_paths)} files to {dest_dir}...")
    for file_path in map(Path, file_paths):
        dest_file = dest_dir / file_path.name
        shutil.copy(file_path, dest_file)
    
    print(f"Successfully copied {len(file_paths)} files to {dest_dir}")