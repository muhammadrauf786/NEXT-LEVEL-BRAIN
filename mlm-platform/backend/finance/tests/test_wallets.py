from django.test import TestCase
from django.contrib.auth import get_user_model
from finance.models import Wallet

class WalletTest(TestCase):
    def test_wallet_creation(self):
        user = get_user_model().objects.create(username="u1")
        w = Wallet.objects.create(user=user, kind="deposit", balance=100)
        self.assertEqual(w.balance, 100)
