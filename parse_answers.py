import csv
import os
import sys
import json
import re

def parse_section(text, header):
    """
    Extracts content for a specific header (e.g., [답변]) from the text.
    Assumes headers are in the format [HeaderName].
    """
    if not text:
        return ""
    
    # Escape brackets for regex
    header_pattern = re.escape(header)
    # Regex to find the content starting after the header
    # and ending before the next header (which starts with [) or end of string.
    # re.DOTALL makes . match newlines
    pattern = f"{header_pattern}\s*(.*?)(?=\n\[|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def parse_answer_details(answer_text):
    """
    Parses the full answer text into specific components.
    """
    return {
        "answer_content": parse_section(answer_text, "[답변]"),
        "relevant_laws": parse_section(answer_text, "[관련 법령]"),
        "relevant_cases": parse_section(answer_text, "[관련 사례/예시]"),
        "precautions": parse_section(answer_text, "[주의사항]")
    }

def parse_tsv(file_path):
    """
    Parses the TSV file and returns a list of dictionaries with structured answer data.
    """
    results = []
    try:
        # Try reading with utf-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Structure the desired output format
                parsed_record = {
                    "query_id": row.get('query_id'),
                    "query": row.get('query'),
                }
                
                # Parse the answer field
                raw_answer = row.get('answer', '')
                parsed_details = parse_answer_details(raw_answer)
                
                # Merge parsed details into the record
                parsed_record.update(parsed_details)
                
                # Exclude timestamp and status (simply don't add them)
                
                results.append(parsed_record)
                
    except UnicodeDecodeError:
        print("UTF-8 decode error. Retrying with utf-8-sig...")
        try:
             with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    parsed_record = {
                        "query_id": row.get('query_id'),
                        "query": row.get('query'),
                    }
                    raw_answer = row.get('answer', '')
                    parsed_record.update(parse_answer_details(raw_answer))
                    results.append(parsed_record)
        except Exception as e:
            print(f"Error parsing file: {e}")
            return []
    except Exception as e:
        print(f"Error parsing file: {e}")
        return []
            
    return results

if __name__ == "__main__":
    # Determine the file path
    # 1. Try relative path first
    relative_path = os.path.join("DATA", "generated_answers", "few_shot", "final_results.tsv")
    
    target_file = relative_path
    if not os.path.exists(target_file):
        # 2. Fallback to absolute path construction based on script location or known workspace
        # Assuming script is in c:\Users\seoki\Desktop\Prj_RAG
        # and file is in DATA/...
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_file = os.path.join(script_dir, "DATA", "generated_answers", "few_shot", "final_results.tsv")

    print(f"Parsing file: {target_file}")
    
    if os.path.exists(target_file):
        data = parse_tsv(target_file)
        print(f"Total records parsed: {len(data)}")
        
        output_file = "parsed_results.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved parsed data to {os.path.abspath(output_file)}")
            
            if data:
                print("Sample record (first item keys):")
                print(list(data[0].keys()))
        except Exception as e:
            print(f"Error saving JSON: {e}")

    else:
        print("File not found.")
