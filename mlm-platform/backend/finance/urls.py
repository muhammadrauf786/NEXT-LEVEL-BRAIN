from django.urls import path
from .views import WalletListView

urlpatterns = [
    path('wallets/', WalletListView.as_view()),
]
