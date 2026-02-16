from rest_framework import generics
from .models import Wallet
from .serializers import WalletSerializer

class WalletListView(generics.ListAPIView):
    serializer_class = WalletSerializer
    def get_queryset(self):
        return Wallet.objects.filter(user=self.request.user)
