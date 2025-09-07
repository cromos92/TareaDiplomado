from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class CustomUserCreationForm(UserCreationForm):
    """Formulario personalizado de registro"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'tu@email.com'
        })
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Tu nombre'
        })
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Tu apellido'
        })
    )

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Personalizar widgets de campos heredados
        self.fields['username'].widget.attrs.update({
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Nombre de usuario'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Contraseña'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Confirmar contraseña'
        })

        # Personalizar help texts
        self.fields['username'].help_text = None
        self.fields['password1'].help_text = None
        self.fields['password2'].help_text = None

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("Ya existe un usuario con este email.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user


class CustomAuthenticationForm(AuthenticationForm):
    """Formulario personalizado de login"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fields['username'].widget.attrs.update({
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Usuario o email'
        })
        self.fields['password'].widget.attrs.update({
            'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent',
            'placeholder': 'Contraseña'
        })

    def clean_username(self):
        username = self.cleaned_data.get('username')
        
        # Permitir login con email o username
        if '@' in username:
            try:
                user = User.objects.get(email=username)
                return user.username
            except User.DoesNotExist:
                pass
        
        return username


class ProfileUpdateForm(forms.ModelForm):
    """Formulario para actualizar perfil"""
    
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')
        widgets = {
            'first_name': forms.TextInput(attrs={
                'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent'
            }),
            'last_name': forms.TextInput(attrs={
                'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-farmacia-500 focus:border-transparent'
            }),
        }

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise ValidationError("Ya existe un usuario con este email.")
        return email
