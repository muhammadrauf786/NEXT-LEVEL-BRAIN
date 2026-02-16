from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    referrer = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="referrals")
    totp_secret = models.CharField(max_length=32, blank=True, null=True)
