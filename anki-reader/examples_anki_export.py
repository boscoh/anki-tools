#!/usr/bin/env python3
"""
Example: Reading Anki cards using anki-export library
Install: uv add anki-export
"""

from pathlib import Path


def example_anki_export():
    """
    anki-export provides simple reading and export capabilities.
    """
    try:
        from anki_export import ApkgReader

        # Find .apkg file
        apkg_files = list(Path('..').glob('*.apkg'))
        if not apkg_files:
            print("No .apkg files found in parent directory")
            return

        apkg_path = apkg_files[0]
        print("=" * 60)
        print("ANKI-EXPORT EXAMPLE")
        print("=" * 60)
        print(f"\nReading: {apkg_path.name}\n")

        with ApkgReader(str(apkg_path)) as apkg:
            # Export returns data structure
            data = apkg.export()

            print(f"Data type: {type(data)}")
            print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")

            # Explore the structure
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"\n{key}:")
                    if isinstance(value, list):
                        print(f"  Type: list with {len(value)} items")
                        if value:
                            print(f"  First item type: {type(value[0])}")
                            if len(value) > 0:
                                print(f"  First item: {value[0]}")
                    else:
                        print(f"  Type: {type(value)}")
                        print(f"  Value: {value}")

            # Try to export to different format
            print("\n" + "=" * 60)
            print("Export capabilities:")
            print("- Can export to Excel with pyexcel-xlsxwx")
            print("- Can export to CSV with pyexcel")
            print("- Provides structured data for custom processing")

    except ImportError as e:
        print(f"anki-export not installed. Install with: uv add anki-export")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    example_anki_export()
