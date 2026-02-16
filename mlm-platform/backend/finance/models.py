from django.db import models
from django.conf import settings
from decimal import Decimal

WALLET_TYPES = [
    ("deposit","deposit"), ("staking","staking"),
    ("daily_profit","daily_profit"),
    ("total_profit","total_profit"),
    ("commission","commission"),
]

class Wallet(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="wallets")
    kind = models.CharField(max_length=32, choices=WALLET_TYPES)
    balance = models.DecimalField(max_digits=28, decimal_places=8, default=Decimal('0'))

    class Meta:
        unique_together = ("user", "kind")
