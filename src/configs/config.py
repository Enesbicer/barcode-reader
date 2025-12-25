import pyzbar.pyzbar as pyzbar


ALLOWED_BARCODES = [
    pyzbar.ZBarSymbol.EAN13,
    pyzbar.ZBarSymbol.EAN8,
    pyzbar.ZBarSymbol.CODE128,
    pyzbar.ZBarSymbol.CODE39,
    pyzbar.ZBarSymbol.UPCA,
    pyzbar.ZBarSymbol.UPCE,
    pyzbar.ZBarSymbol.I25
]