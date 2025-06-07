#!/usr/bin/env python3

import sys
import traceback

def tiered_mileage(miles: float) -> float:
    if miles <= 100:
        return miles * 0.50
    elif miles <= 300:
        return 100 * 0.50 + (miles - 100) * 0.30
    else:
        return 100 * 0.50 + 200 * 0.30 + (miles - 300) * 0.10

def adjusted_receipts(receipts: float) -> float:
    """Apply soft caps with diminishing returns to receipts."""
    if receipts <= 800:
        return receipts
    elif receipts <= 1500:
        return 800 + (receipts - 800) * 0.5
    else:
        return 800 + 700 * 0.5 + (receipts - 1500) * 0.1

def basic_reimbursement(days: int, miles: float, receipts: float) -> float:
    PER_DIEM = 150 * (0.85 ** (days - 1))
    
    mileage_reimbursement = tiered_mileage(miles)
    reimbursement = (
        PER_DIEM * days +
        mileage_reimbursement +
        adjusted_receipts(receipts)
    )

    return round(reimbursement, 2)

if __name__ == "__main__":
    try:
        if len(sys.argv) != 4:
            print("Usage: python3 calculate_reimbursement.py <days> <miles> <receipts>")
            sys.exit(1)
            
        # Print debug info to stderr
        print(f"Debug - Input: days={sys.argv[1]}, miles={sys.argv[2]}, receipts={sys.argv[3]}", file=sys.stderr)
        
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = basic_reimbursement(days, miles, receipts)
        print(result)  # Print only the number to stdout
        
    except Exception as e:
        # Print full error details to stderr
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1) 