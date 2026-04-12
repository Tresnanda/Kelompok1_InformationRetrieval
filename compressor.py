from typing import List, Dict, Tuple, Set

class VBEncoder:
    """Variable Byte Encoding for integer compression"""
    
    @staticmethod
    def encode(number: int) -> bytes:
        """Encode a single integer using VB encoding"""
        bytes_list = []
        while True:
            bytes_list.insert(0, number % 128)
            if number < 128:
                break
            number //= 128
        bytes_list[-1] += 128  # Set continuation bit on last byte
        return bytes(bytes_list)
    
    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        """Encode a list of integers"""
        return b''.join(VBEncoder.encode(n) for n in numbers)
    
    @staticmethod
    def decode(byte_stream: bytes) -> List[int]:
        """Decode VB encoded bytes back to integers"""
        numbers = []
        current = 0
        for byte in byte_stream:
            if byte < 128:
                current = 128 * current + byte
            else:
                current = 128 * current + (byte - 128)
                numbers.append(current)
                current = 0
        return numbers


class GapEncoder:
    """Gap encoding for docID sequences"""
    
    @staticmethod
    def encode(doc_ids: List[int]) -> List[int]:
        """Convert absolute docIDs to gaps"""
        if not doc_ids:
            return []
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gaps.append(doc_ids[i] - doc_ids[i-1])
        return gaps
    
    @staticmethod
    def decode(gaps: List[int]) -> List[int]:
        """Convert gaps back to absolute docIDs"""
        if not gaps:
            return []
        doc_ids = [gaps[0]]
        for i in range(1, len(gaps)):
            doc_ids.append(doc_ids[i-1] + gaps[i])
        return doc_ids