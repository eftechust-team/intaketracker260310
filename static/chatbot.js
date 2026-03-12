// Chatbot JavaScript - handles food input, API calls, and UI updates

class NutritionChatbot {
    constructor() {
        this.currentUserId = null;
        this.currentUserName = 'Guest';
        this.allUsers = [];
        this.dailyNutrition = {
            carbs: 0,
            protein: 0,
            fat: 0,
            calories: 0
        };
        this.userInfo = {};
        this.userRecordId = null;
        this.conversationHistory = [];
        this._recipePlanners = {};
        this.init();
    }

    init() {
        // DOM elements
        this.messagesContainer = document.getElementById('messages-container');
        this.foodInput = document.getElementById('food-input');
        this.foodInputForm = document.getElementById('food-input-form');
        this.userInfoForm = document.getElementById('user-info-form');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.undoBtn = document.getElementById('undo-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.saveUserInfoBtn = document.getElementById('save-user-info-btn');
        this.switchUserBtn = document.getElementById('switch-user-btn');
        this.currentUserDisplay = document.getElementById('current-user-name');
        this.calendarBtn = document.getElementById('calendar-btn');
        this.calendarModal = document.getElementById('calendar-modal');
        this.closeCalendarModalBtn = document.getElementById('close-calendar-modal');
        this.clearConfirmModal = document.getElementById('clear-confirm-modal');
        this.clearConfirmBtn = document.getElementById('clear-confirm-btn');
        this.clearCancelBtn = document.getElementById('clear-cancel-btn');
        this.userModal = document.getElementById('user-modal');
        this.closeUserModalBtn = document.getElementById('close-user-modal');
        this.createUserBtn = document.getElementById('create-user-btn');
        this.newUserNameInput = document.getElementById('new-user-name');
        this.userList = document.getElementById('user-list');
        this.defaultMessagesHtml = this.messagesContainer ? this.messagesContainer.innerHTML : '';

        // Counters (may not exist if stat boxes were removed)
        this.carbsDisplay = document.getElementById('carbs-total');
        this.proteinDisplay = document.getElementById('protein-total');
        this.fatDisplay = document.getElementById('fat-total');
        this.caloriesDisplay = document.getElementById('calories-total');
        this.intakeProgress = document.getElementById('intake-progress');

        // Event listeners
        this.foodInputForm.addEventListener('submit', (e) => this.handleFoodInput(e));
        this.userInfoForm.addEventListener('change', () => this.updateUserInfo());
        if (this.saveUserInfoBtn) {
            this.saveUserInfoBtn.addEventListener('click', () => this.handleSaveUserInfo());
        }
        this.analyzeBtn.addEventListener('click', () => this.handleAnalyze());
        this.undoBtn.addEventListener('click', () => this.undoLast());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        this.clearConfirmBtn.addEventListener('click', () => this._confirmClear());
        this.clearCancelBtn.addEventListener('click', () => this._closeClearModal());
        this.clearConfirmModal.addEventListener('click', (e) => { if (e.target === this.clearConfirmModal) this._closeClearModal(); });
        this.switchUserBtn.addEventListener('click', () => this.openUserModal());
        this.closeUserModalBtn.addEventListener('click', () => this.closeUserModal());
        this.createUserBtn.addEventListener('click', () => this.handleCreateUser());
        this.calendarBtn.addEventListener('click', () => this.openCalendar());
        this.closeCalendarModalBtn.addEventListener('click', () => this.closeCalendar());
        this.calendarModal.addEventListener('click', (e) => {
            if (e.target === this.calendarModal) this.closeCalendar();
        });
        this.userModal.addEventListener('click', (e) => {
            if (e.target === this.userModal) this.closeUserModal();
        });

        // Load saved data and initialize users
        this.loadSavedData();
        this.loadAllUsers();
        
        // Update button status based on loaded data
        this.updateAnalyzeButtonStatus();
    }

    async loadAllUsers() {
        try {
            const response = await fetch('/api/user-records');
            if (response.ok) {
                const data = await response.json();
                this.allUsers = data.records || [];
                console.log('[MultiUser] Loaded users:', this.allUsers);
                
                // If no current user, show modal to create one
                if (!this.currentUserId && this.allUsers.length === 0) {
                    setTimeout(() => this.openUserModal(), 500);
                }
            }
        } catch (err) {
            console.error('[MultiUser] Failed to load users:', err);
        }
    }

    loadSavedData() {
        const savedUserRecordId = localStorage.getItem('userRecordId');
        const savedCurrentUserId = localStorage.getItem('currentUserId');
        const savedCurrentUserName = localStorage.getItem('currentUserName');

        if (savedUserRecordId) {
            this.userRecordId = savedUserRecordId || null;
        }

        if (savedCurrentUserId) {
            this.currentUserId = savedCurrentUserId;
        }

        if (savedCurrentUserName) {
            this.currentUserName = savedCurrentUserName;
            this.updateCurrentUserDisplay();
        }

        this.loadCurrentUserState();
    }

    saveData() {
        localStorage.setItem('userRecordId', this.userRecordId || '');
        localStorage.setItem('currentUserId', this.currentUserId || '');
        localStorage.setItem('currentUserName', this.currentUserName || 'Guest');
        this.saveCurrentUserState();
    }

    getUserStateKey(userId) {
        return `userState:${userId || 'guest'}`;
    }

    saveCurrentUserState() {
        // Only save state for real (non-guest) users
        if (!this.currentUserId) return;
        const key = this.getUserStateKey(this.currentUserId);
        const state = {
            dailyNutrition: this.dailyNutrition,
            userInfo: this.userInfo,
            conversationHistory: this.conversationHistory,
            messagesHtml: this.messagesContainer ? this.messagesContainer.innerHTML : ''
        };
        localStorage.setItem(key, JSON.stringify(state));
    }

    loadCurrentUserState() {
        const key = this.getUserStateKey(this.currentUserId);
        const raw = localStorage.getItem(key);

        // Defaults for new users without saved state
        this.dailyNutrition = { carbs: 0, protein: 0, fat: 0, calories: 0 };
        this.userInfo = {};
        this.conversationHistory = [];

        if (raw) {
            try {
                const state = JSON.parse(raw);
                this.dailyNutrition = state.dailyNutrition || { carbs: 0, protein: 0, fat: 0, calories: 0 };
                // Ensure legacy saved states without calories still work
                if (this.dailyNutrition.calories == null) this.dailyNutrition.calories = 0;
                this.userInfo = state.userInfo || {};
                this.conversationHistory = state.conversationHistory || [];
                if (this.messagesContainer && state.messagesHtml) {
                    this.messagesContainer.innerHTML = state.messagesHtml;
                    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
                }
            } catch (err) {
                console.error('[MultiUser] Failed to parse user state:', err);
            }
        } else if (this.messagesContainer) {
            // New user with no saved state: show default intro, not previous user's messages.
            this.messagesContainer.innerHTML = this.defaultMessagesHtml || '';
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }

        this.updateDisplay();
        this.populateUserForm();
        this.updateAnalyzeButtonStatus();
    }



    updateUserInfo() {
        const formData = new FormData(this.userInfoForm);
        this.userInfo = {
            gender: formData.get('gender'),
            age: formData.get('age'),
            height: formData.get('height'),
            weight: formData.get('weight'),
            activity: formData.get('activity'),
            diet: formData.get('diet'),
            preference: formData.get('preference')
        };
        this.saveData();
        this.updateAnalyzeButtonStatus();
        if (this._profileIsFilled()) this._updateProgressBars();
    }

    async handleSaveUserInfo() {
        // Collect latest form values then persist
        this.updateUserInfo();

        const btn = this.saveUserInfoBtn;
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'Saving...';
        }

        try {
            if (this.currentUserId) {
                await this.saveUserInfoToBackend();
            }
            if (btn) {
                btn.textContent = '✓ Saved';
                btn.style.background = 'var(--success, #22c55e)';
                setTimeout(() => {
                    btn.textContent = 'Save';
                    btn.style.background = '';
                    btn.disabled = false;
                }, 1800);
            }
        } catch (err) {
            console.error('[UserInfo] Save failed:', err);
            if (btn) {
                btn.textContent = 'Save failed';
                setTimeout(() => {
                    btn.textContent = 'Save';
                    btn.disabled = false;
                }, 2000);
            }
        }
    }

    async saveUserInfoToBackend() {
        if (!this.currentUserId) return;
        
        try {
            await fetch('/api/user-records', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this.currentUserId,
                    user_info: { ...this.userInfo, name: this.currentUserName }
                })
            });
            console.log('[UserInfo] Auto-saved user info to backend');
        } catch (err) {
            console.error('[UserInfo] Failed to save user info:', err);
        }
    }

    populateUserForm() {
        this.clearUserForm();
        if (this.userInfo.gender) {
            document.querySelector(`input[name="gender"][value="${this.userInfo.gender}"]`).checked = true;
        }
        if (this.userInfo.age) document.querySelector('input[name="age"]').value = this.userInfo.age;
        if (this.userInfo.height) document.querySelector('input[name="height"]').value = this.userInfo.height;
        if (this.userInfo.weight) document.querySelector('input[name="weight"]').value = this.userInfo.weight;
        if (this.userInfo.activity) document.querySelector('select[name="activity"]').value = this.userInfo.activity;
        if (this.userInfo.diet) document.querySelector('select[name="diet"]').value = this.userInfo.diet;
        if (this.userInfo.preference) {
            document.querySelector(`input[name="preference"][value="${this.userInfo.preference}"]`).checked = true;
        }
    }

    clearUserForm() {
        this.userInfoForm.reset();
    }

    updateCurrentUserDisplay() {
        if (this.currentUserDisplay) {
            this.currentUserDisplay.textContent = this.currentUserName || 'Guest';
        }
    }

    openUserModal() {
        this.userModal.style.display = 'flex';
        this.renderUserList();
    }

    closeUserModal() {
        this.userModal.style.display = 'none';
    }

    renderUserList() {
        this.userList.innerHTML = '';

        if (this.allUsers.length === 0) {
            this.userList.innerHTML = '<div class="no-users-message">No users yet — add one below.</div>';
            return;
        }

        this.allUsers.forEach(user => {
            const name = user.user_info?.name || user.id;
            const isActive = user.id === this.currentUserId;

            const userItem = document.createElement('div');
            userItem.className = 'user-card' + (isActive ? ' user-card-active' : '');

            // Avatar circle with initials
            const avatar = document.createElement('div');
            avatar.className = 'user-avatar';
            avatar.textContent = name.charAt(0).toUpperCase();

            // Name + status
            const info = document.createElement('div');
            info.className = 'user-card-info';

            const nameSpan = document.createElement('span');
            nameSpan.className = 'user-card-name';
            nameSpan.textContent = name;
            nameSpan.dataset.userId = user.id;

            info.appendChild(nameSpan);
            if (isActive) {
                const badge = document.createElement('span');
                badge.className = 'user-active-badge';
                badge.textContent = 'Active';
                info.appendChild(badge);
            }

            // Action buttons
            const actions = document.createElement('div');
            actions.className = 'user-card-actions';

            if (!isActive) {
                const switchBtn = document.createElement('button');
                switchBtn.className = 'uca-btn uca-switch';
                switchBtn.title = 'Switch to this user';
                switchBtn.innerHTML = '&#x21C4;';
                switchBtn.addEventListener('click', () => this.switchToUser(user));
                actions.appendChild(switchBtn);
            }

            const editBtn = document.createElement('button');
            editBtn.className = 'uca-btn uca-edit';
            editBtn.title = 'Rename';
            editBtn.innerHTML = '&#x270E;';
            editBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.startInlineUserNameEdit(user.id, nameSpan, userItem);
            });

            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'uca-btn uca-delete';
            deleteBtn.title = 'Delete user';
            deleteBtn.innerHTML = '&#x1F5D1;';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteUser(user.id);
            });

            actions.appendChild(editBtn);
            actions.appendChild(deleteBtn);

            userItem.appendChild(avatar);
            userItem.appendChild(info);
            userItem.appendChild(actions);
            this.userList.appendChild(userItem);
        });
    }

    startInlineUserNameEdit(userId, userNameSpan, userItem) {
        const user = this.allUsers.find(u => u.id === userId);
        if (!user || !userNameSpan || !userItem) return;

        // Prevent duplicate inline editors in same row
        if (userItem.querySelector('.inline-rename-wrapper')) return;

        const currentName = user.user_info?.name || userId;

        // infoDiv is the direct child of userItem that contains nameSpan
        const infoDiv = userNameSpan.closest('.user-card-info') || userNameSpan.parentNode;
        const actionsDiv = userItem.querySelector('.user-card-actions');

        // Hide actions column; wrapper will span both name + actions columns via CSS grid-column
        if (actionsDiv) actionsDiv.style.display = 'none';

        const wrapper = document.createElement('div');
        wrapper.className = 'inline-rename-wrapper';

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'inline-rename-input';
        input.value = currentName;

        const saveBtn = document.createElement('button');
        saveBtn.type = 'button';
        saveBtn.className = 'inline-rename-save';
        saveBtn.textContent = 'Save';

        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button';
        cancelBtn.className = 'inline-rename-cancel';
        cancelBtn.textContent = 'Cancel';

        wrapper.appendChild(input);
        wrapper.appendChild(saveBtn);
        wrapper.appendChild(cancelBtn);

        // Replace the info div (direct child of userItem) with the wrapper
        userItem.replaceChild(wrapper, infoDiv);
        input.focus();
        input.select();

        const cancelEdit = () => {
            if (wrapper.parentNode === userItem) {
                userItem.replaceChild(infoDiv, wrapper);
            }
            if (actionsDiv) actionsDiv.style.display = '';
        };

        saveBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            await this.editUserName(userId, input.value.trim());
        });

        cancelBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            cancelEdit();
        });

        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                await this.editUserName(userId, input.value.trim());
            }
            if (e.key === 'Escape') {
                e.preventDefault();
                cancelEdit();
            }
        });
    }

    async editUserName(userId, newName) {
        const user = this.allUsers.find(u => u.id === userId);
        if (!user) return;

        const trimmedName = (newName || '').trim();
        if (!trimmedName) return;

        try {
            const updatedUserInfo = { ...(user.user_info || {}), name: trimmedName };
            const response = await fetch('/api/user-records', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    user_info: updatedUserInfo
                })
            });

            if (!response.ok) {
                alert('Failed to update user name. Please try again.');
                return;
            }

            user.user_info = updatedUserInfo;

            if (this.currentUserId === userId) {
                this.currentUserName = trimmedName;
                this.updateCurrentUserDisplay();
                this.saveData();
            }

            this.renderUserList();
            this.addMessage(`✏️ User renamed to "${trimmedName}".`, 'bot');
        } catch (err) {
            console.error('[MultiUser] Failed to edit user name:', err);
            alert('Error updating user name.');
        }
    }

    async switchToUser(user) {
        console.log('[MultiUser] Switching to user:', user);

        // Save current user's daily intake if we have a current user
        if (this.currentUserId && this.userRecordId) {
            try {
                await this.saveDailyIntakeToBackend();
            } catch (err) {
                console.error('[MultiUser] Failed to save current user intake:', err);
            }
        }

        // Persist current user's local state before switching
        this.saveCurrentUserState();

        // Switch to new user
        this.currentUserId = user.id;
        this.currentUserName = user.user_info?.name || user.id;
        this.userRecordId = user.id;

        // Load selected user's local state (chat, intake, profile)
        this.loadCurrentUserState();

        // If no local profile exists yet, initialize from backend profile
        const hasLocalProfile = this.userInfo && Object.keys(this.userInfo).length > 0;
        if (!hasLocalProfile && user.user_info) {
            this.userInfo = { ...user.user_info };
            this.populateUserForm();
            this.saveCurrentUserState();
        }

        // Save to localStorage
        this.saveData();
        this.updateCurrentUserDisplay();
        this.renderUserList();
        this.closeUserModal();

        this.addMessage(`👤 Switched to ${this.currentUserName}'s account.`, 'bot');
    }

    async handleCreateUser() {
        const userName = this.newUserNameInput.value.trim();
        if (!userName) {
            alert('Please enter a name for the new user.');
            return;
        }

        try {
            const response = await fetch('/api/user-records', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: userName })
            });

            if (!response.ok) {
                alert('Failed to create user. Please try again.');
                return;
            }

            const data = await response.json();
            console.log('[MultiUser] Created new user:', data);

            // Add to local users list
            this.allUsers.push(data.user);

            // Reload the user list
            this.renderUserList();

            // Clear input
            this.newUserNameInput.value = '';

            this.addMessage(`✅ Created new user "${userName}".`, 'bot');
        } catch (err) {
            console.error('[MultiUser] Failed to create user:', err);
            alert('Error creating user:', err.message);
        }
    }

    async deleteUser(userId) {
        const user = this.allUsers.find(u => u.id === userId);
        if (!user) return;

        const ok = confirm(`Delete user "${user.user_info?.name || userId}"?`);
        if (!ok) return;

        try {
            const response = await fetch(`/api/user-records/${userId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                alert('Failed to delete user.');
                return;
            }

            // Remove from local list
            this.allUsers = this.allUsers.filter(u => u.id !== userId);

            // If deleted user was current, reset to first user or guest
            if (this.currentUserId === userId) {
                if (this.allUsers.length > 0) {
                    await this.switchToUser(this.allUsers[0]);
                } else {
                    this.currentUserId = null;
                    this.currentUserName = 'Guest';
                    this.userRecordId = null;
                    this.dailyNutrition = { carbs: 0, protein: 0, fat: 0, calories: 0 };
                    this.updateDisplay();
                    this.saveData();
                    this.updateCurrentUserDisplay();
                }
            }

            this.renderUserList();
            this.addMessage(`🗑️ Deleted user.`, 'bot');
        } catch (err) {
            console.error('[MultiUser] Failed to delete user:', err);
            alert('Error deleting user:', err.message);
        }
    }

    async saveDailyIntakeToBackend() {
        if (!this.currentUserId) return;

        try {
            // Attach recommended values if the user has run a recommendation this session
            let recommended = {};
            const lastRec = sessionStorage.getItem('lastRecommendation');
            if (lastRec) {
                try {
                    const rec = JSON.parse(lastRec);
                    recommended = {
                        calories: rec.calories,
                        carbs: rec.carbohydrate_intake,
                        protein: rec.protein_intake,
                        fat: rec.fat_intake
                    };
                } catch (_) {}
            }

            const response = await fetch(`/api/daily-intake/${this.currentUserId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    daily_nutrition: this.dailyNutrition,
                    recommended,
                    timestamp: new Date().toISOString()
                })
            });

            if (response.ok) {
                console.log('[MultiUser] Saved daily intake for user:', this.currentUserId);
            }
        } catch (err) {
            console.error('[MultiUser] Failed to save daily intake:', err);
        }
    }

    async handleFoodInput(e) {
        e.preventDefault();

        const foodInput = this.foodInput.value.trim();
        if (!foodInput) return;

        // Add user message to chat
        this.addMessage(foodInput, 'user');
        this.foodInput.value = '';

        // Check for direct nutrition input (e.g., "50g carbs", "-20g fat", "1000kcal", "-100 kcal")
        const directMacroMatch = foodInput.match(/^([+-]?\d+(?:\.\d+)?)\s*g?\s*(carb|carbon|carbohydrate|protein|fat)s?$/i);
        const directCalorieMatch = foodInput.match(/^([+-]?\d+(?:\.\d+)?)\s*(kcal|cal|calorie|calories)$/i);
        if (directMacroMatch || directCalorieMatch) {
            const amount = parseFloat((directMacroMatch || directCalorieMatch)[1]);
            const macroType = directMacroMatch ? directMacroMatch[2].toLowerCase() : 'calories';
            
            let macroName = '';
            let nutrition = { carbs: 0, protein: 0, fat: 0, calories: 0 };
            
            if (macroType === 'carb' || macroType === 'carbon' || macroType === 'carbohydrate') {
                nutrition.carbs = amount;
                macroName = 'Carbohydrates';
            } else if (macroType === 'protein') {
                nutrition.protein = amount;
                macroName = 'Protein';
            } else if (macroType === 'fat') {
                nutrition.fat = amount;
                macroName = 'Fat';
            } else {
                nutrition.calories = amount;
                macroName = 'Calories';
            }
            
            this.addFoodToDaily(nutrition);
            
            const actionText = amount >= 0 ? 'added' : 'removed';
            const amountUnit = macroName === 'Calories' ? 'kcal' : 'g';
            const responseMsg = `✅ <strong>${macroName}</strong>
            
<div class="food-item-display">
    <div class="food-name">${Math.abs(amount)}${amountUnit} ${actionText} directly</div>
    <div class="nutrition-row">
        <span>Carbs:</span>
        <span>${nutrition.carbs}g</span>
    </div>
    <div class="nutrition-row">
        <span>Protein:</span>
        <span>${nutrition.protein}g</span>
    </div>
    <div class="nutrition-row">
        <span>Fat:</span>
        <span>${nutrition.fat}g</span>
    </div>
    <div class="nutrition-row">
        <span>Calories:</span>
        <span>${nutrition.calories} kcal</span>
    </div>
</div>

Your daily totals have been updated. Keep tracking!`;
            
            this.addMessage(responseMsg, 'bot');
            
            // Add to conversation history
            this.conversationHistory.push({
                input: foodInput,
                nutrition: { ...nutrition, food_name: macroName, quantity: amount, unit: macroName === 'Calories' ? 'kcal' : 'g' },
                timestamp: new Date()
            });
            
            this.updateAnalyzeButtonStatus();
            this.saveData();
            return;
        }

        // Show loading state — update text if fallback sources are used
        const loadingMsg = this.addMessage('🔍 Searching CSV database...', 'bot', true);

        // If the request takes longer, CSV missed and we're hitting USDA or Doubao
        const usdaTimer = setTimeout(() => {
            this.updateMessage(loadingMsg, '🔍 Not in CSV, checking USDA API...');
        }, 900);
        const doubaoTimer = setTimeout(() => {
            this.updateMessage(loadingMsg, '🤖 Not in USDA, querying Doubao AI...');
        }, 3500);

        try {
            // Call the API
            const response = await fetch('/api/search-food', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ food_input: foodInput })
            });

            const result = await response.json();
            clearTimeout(usdaTimer);
            clearTimeout(doubaoTimer);

            if (!response.ok) {
                // More helpful error messages
                let errorMsg = result.error || 'Unable to find food information';
                if (errorMsg.includes('No foods found')) {
                    errorMsg += '\n\nTry being more specific (e.g., "chicken breast" instead of just "chicken")';
                }
                this.updateMessage(loadingMsg, `❌ ${errorMsg}`);
                return;
            }

            // Extract nutrition data
            const nutrition = result.nutrition;
            const individualFoods = result.individual_foods || [];
            
            // Check if nutrition data is actually valid (not all zeros across macros and calories)
            if ((nutrition.carbs || 0) === 0 && (nutrition.protein || 0) === 0 && (nutrition.fat || 0) === 0 && (nutrition.calories || 0) === 0) {
                this.updateMessage(loadingMsg, `⚠️ Found foods but nutrition data appears incomplete.\n\nThis might be a data limitation. Try a different food or variation.`);
                return;
            }
            
            this.addFoodToDaily(nutrition);

            // Generate friendly response showing all foods if multiple
            const responseMsg = this.generateFoodResponse(nutrition, individualFoods);
            this.updateMessage(loadingMsg, responseMsg);

            // Add to conversation history
            this.conversationHistory.push({
                input: foodInput,
                nutrition: nutrition,
                individualFoods: individualFoods,
                timestamp: new Date()
            });

            // Auto-update analyze button status
            this.updateAnalyzeButtonStatus();
            
            this.saveData();

        } catch (error) {
            clearTimeout(usdaTimer);
            clearTimeout(doubaoTimer);
            console.error('Error:', error);
            this.updateMessage(loadingMsg, '❌ Error processing food. Please try again.');
        }
    }

    _sourceLabel(source) {
        if (!source || source === 'CSV') return 'CSV database';
        if (source.includes('USDA')) return 'USDA API';
        if (source.includes('Doubao')) return 'Doubao AI';
        return source;
    }

    generateFoodResponse(nutrition, individualFoods = []) {
        let foodsDisplay = '';

        // Show individual foods if multiple
        if (individualFoods.length > 1) {
            foodsDisplay = '<div class="foods-list">';
            individualFoods.forEach(food => {
                const src = this._sourceLabel(food.source);
                foodsDisplay += `
                <div class="food-item-display" style="margin-bottom: 12px;">
                    <div class="food-name">${food.quantity}${food.unit} ${food.food_name}</div>
                    <div class="nutrition-row">
                        <span>Calories:</span>
                        <span>${food.calories ?? 0} kcal</span>
                    </div>
                    <div class="nutrition-row">
                        <span>Carbs:</span>
                        <span>${food.carbs}g</span>
                    </div>
                    <div class="nutrition-row">
                        <span>Protein:</span>
                        <span>${food.protein}g</span>
                    </div>
                    <div class="nutrition-row">
                        <span>Fat:</span>
                        <span>${food.fat}g</span>
                    </div>
                    <div class="food-source">Source: ${src}</div>
                </div>`;
            });
            foodsDisplay += '</div>';

            return `
✅ <strong>Added ${individualFoods.length} foods</strong>

${foodsDisplay}

<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee;">
    <strong>Total Added:</strong>
    <div class="nutrition-row">
        <span>Calories:</span>
        <span>${nutrition.calories ?? 0} kcal</span>
    </div>
    <div class="nutrition-row">
        <span>Carbs:</span>
        <span>${nutrition.carbs}g</span>
    </div>
    <div class="nutrition-row">
        <span>Protein:</span>
        <span>${nutrition.protein}g</span>
    </div>
    <div class="nutrition-row">
        <span>Fat:</span>
        <span>${nutrition.fat}g</span>
    </div>
</div>

Your daily totals have been updated. Keep tracking!
            `;
        } else if (individualFoods.length === 1) {
            const food = individualFoods[0];
            const src = this._sourceLabel(food.source || nutrition.source);
            return `
✅ <strong>${food.food_name}</strong>

<div class="food-item-display">
    <div class="food-name">${food.quantity}${food.unit} added</div>
    <div class="nutrition-row">
        <span>Calories:</span>
        <span>${food.calories ?? 0} kcal</span>
    </div>
    <div class="nutrition-row">
        <span>Carbs:</span>
        <span>${food.carbs}g</span>
    </div>
    <div class="nutrition-row">
        <span>Protein:</span>
        <span>${food.protein}g</span>
    </div>
    <div class="nutrition-row">
        <span>Fat:</span>
        <span>${food.fat}g</span>
    </div>
    <div class="food-source">Source: ${src}</div>
</div>

Your daily totals have been updated. Keep tracking!
            `;
        } else {
            // Fallback to old format if no individual foods
            return `
✅ <strong>Food Added</strong>

<div class="food-item-display">
    <div class="nutrition-row">
        <span>Calories:</span>
        <span>${nutrition.calories ?? 0} kcal</span>
    </div>
    <div class="nutrition-row">
        <span>Carbs:</span>
        <span>${nutrition.carbs}g</span>
    </div>
    <div class="nutrition-row">
        <span>Protein:</span>
        <span>${nutrition.protein}g</span>
    </div>
    <div class="nutrition-row">
        <span>Fat:</span>
        <span>${nutrition.fat}g</span>
    </div>
</div>

Your daily totals have been updated. Keep tracking!
            `;
        }
    }

    addFoodToDaily(nutrition) {
        this.dailyNutrition.carbs += nutrition.carbs || 0;
        this.dailyNutrition.protein += nutrition.protein || 0;
        this.dailyNutrition.fat += nutrition.fat || 0;
        this.dailyNutrition.calories += nutrition.calories || 0;

        // Round to 2 decimal places
        this.dailyNutrition.carbs = Math.round(this.dailyNutrition.carbs * 100) / 100;
        this.dailyNutrition.protein = Math.round(this.dailyNutrition.protein * 100) / 100;
        this.dailyNutrition.fat = Math.round(this.dailyNutrition.fat * 100) / 100;
        this.dailyNutrition.calories = Math.round(this.dailyNutrition.calories * 10) / 10;

        this.updateDisplay();
        this.saveData();
    }

    updateDisplay() {
        if (this.caloriesDisplay) this.caloriesDisplay.textContent = Math.round(this.dailyNutrition.calories || 0);
        if (this.carbsDisplay) this.carbsDisplay.textContent = this.dailyNutrition.carbs.toFixed(1);
        if (this.proteinDisplay) this.proteinDisplay.textContent = this.dailyNutrition.protein.toFixed(1);
        if (this.fatDisplay) this.fatDisplay.textContent = this.dailyNutrition.fat.toFixed(1);
        this._updateProgressBars();
    }

    _profileIsFilled() {
        const u = this.userInfo;
        return u && u.gender != null && u.gender !== '' &&
            parseFloat(u.age) > 0 && parseFloat(u.height) > 0 && parseFloat(u.weight) > 0 &&
            u.activity !== '' && u.activity != null &&
            u.diet !== '' && u.diet != null;
    }

    _calcTargets() {
        // Recompute targets from profile (mirrors backend calculate_rmr / calculate_daily_calories)
        const u = this.userInfo;
        const weight = parseFloat(u.weight) || 70;
        const height = parseFloat(u.height) || 170;
        const age    = parseFloat(u.age)    || 25;
        const gender = parseInt(u.gender)   || 0;
        const activity = parseInt(u.activity) || 2;
        const diet   = parseInt(u.diet)     || 0;

        let rmr = gender === 0
            ? 9.99 * weight + 6.25 * height - 4.92 * age + 5
            : 9.99 * weight + 6.25 * height - 4.92 * age - 161;
        const actFactors = [1.2, 1.375, 1.55, 1.725];
        const calories = rmr * (actFactors[activity] || 1.55);

        // diet plan macros (fraction of calories)
        const dietScales = [
            [0.50, 0.20, 0.30], // balanced
            [0.60, 0.20, 0.20], // low fat
            [0.20, 0.30, 0.50], // low carb
            [0.28, 0.39, 0.33], // high protein
        ];
        const [cf, pf, ff] = dietScales[diet] || dietScales[0];
        return {
            calories: Math.round(calories),
            carbs:   Math.round(calories * cf / 4.1),
            protein: Math.round(calories * pf / 4.1),
            fat:     Math.round(calories * ff / 8.8),
        };
    }

    _updateProgressBars() {
        if (!this.intakeProgress) return;

        // Check for targets from last recommendation first, fall back to computed
        let targets = null;
        const lastRec = sessionStorage.getItem('lastRecommendation');
        if (lastRec) {
            try {
                const rec = JSON.parse(lastRec);
                if (rec && rec.calories) {
                    targets = {
                        calories: Math.round(rec.calories),
                        carbs:    Math.round(rec.carbohydrate_intake),
                        protein:  Math.round(rec.protein_intake),
                        fat:      Math.round(rec.fat_intake),
                    };
                }
            } catch (_) {}
        }
        if (!targets && this._profileIsFilled()) {
            targets = this._calcTargets();
        }

        const bars = [
            { id: 'calories', intake: Math.round(this.dailyNutrition.calories || 0), unit: 'kcal', target: targets?.calories },
            { id: 'carbs',    intake: parseFloat(this.dailyNutrition.carbs.toFixed(1)),   unit: 'g', target: targets?.carbs },
            { id: 'protein',  intake: parseFloat(this.dailyNutrition.protein.toFixed(1)), unit: 'g', target: targets?.protein },
            { id: 'fat',      intake: parseFloat(this.dailyNutrition.fat.toFixed(1)),     unit: 'g', target: targets?.fat },
        ];

        bars.forEach(({ id, intake, unit, target }) => {
            const barEl  = document.getElementById(`${id}-bar`);
            const textEl = document.getElementById(`${id}-progress-text`);
            if (!barEl || !textEl) return;

            if (target) {
                // Profile filled: split-bar — base portion + excess portion
                const rawPct  = Math.round(intake / target * 100);
                const excessEl = document.getElementById(`${id}-bar-excess`);
                textEl.textContent = `${intake} / ${target} ${unit}`;
                barEl.classList.remove('bar-low', 'bar-mid', 'bar-over', 'bar-danger', 'has-excess');

                if (rawPct <= 100) {
                    barEl.style.width = rawPct + '%';
                    if (excessEl) { excessEl.style.width = '0%'; excessEl.classList.remove('pulsing'); }
                    if (rawPct <= 80) barEl.classList.add('bar-low');
                    else              barEl.classList.add('bar-mid');
                } else {
                    // Split: base occupies its proportional share, excess takes the rest
                    const baseW   = (100 / rawPct * 100).toFixed(2) + '%';
                    const excessW = ((rawPct - 100) / rawPct * 100).toFixed(2) + '%';
                    barEl.style.width = baseW;
                    barEl.classList.add('has-excess');
                    if (rawPct <= 120) barEl.classList.add('bar-over');
                    else               barEl.classList.add('bar-danger');
                    if (excessEl) { excessEl.style.width = excessW; excessEl.classList.add('pulsing'); }
                }
            } else {
                // No profile: just show raw intake, empty bar
                const excessEl = document.getElementById(`${id}-bar-excess`);
                barEl.style.width = '0%';
                textEl.textContent = `${intake} ${unit}`;
                barEl.classList.remove('bar-low', 'bar-mid', 'bar-over', 'bar-danger', 'has-excess');
                if (excessEl) { excessEl.style.width = '0%'; excessEl.classList.remove('pulsing'); }
            }
        });
    }

    addMessage(text, sender = 'bot', isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        if (isLoading) messageDiv.classList.add('loading');

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isLoading) {
            contentDiv.innerHTML = `<div class="loading-spinner"></div> ${text}`;
        } else {
            contentDiv.innerHTML = text;
        }

        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;

        // Persist message history only for identified users
        if (this.currentUserId) {
            this.saveCurrentUserState();
        }

        return messageDiv;
    }

    updateMessage(messageElement, newText) {
        const contentDiv = messageElement.querySelector('.message-content');
        contentDiv.innerHTML = newText;
        messageElement.classList.remove('loading');
        if (this.currentUserId) {
            this.saveCurrentUserState();
        }
    }

    async handleAnalyze() {
        // Validate user info - provide specific guidance
        const missingFields = [];
        
        if (!this.userInfo.gender) missingFields.push('Gender');
        if (!this.userInfo.age) missingFields.push('Age');
        if (!this.userInfo.height) missingFields.push('Height');
        if (!this.userInfo.weight) missingFields.push('Weight');
        if (!this.userInfo.activity || this.userInfo.activity === '') missingFields.push('Activity Level');
        if (!this.userInfo.diet || this.userInfo.diet === '') missingFields.push('Diet Plan');
        if (!this.userInfo.preference || this.userInfo.preference === '') missingFields.push('Food Preference');

        if (missingFields.length > 0) {
            this.addMessage(`📋 Missing information needed:\n• ${missingFields.join('\n• ')}\n\nPlease fill in the form on the left, then try again.`, 'bot');
            return;
        }

        // Check if at least some food has been logged
        if (this.dailyNutrition.carbs === 0 && this.dailyNutrition.protein === 0 && this.dailyNutrition.fat === 0) {
            this.addMessage('🍽️ No food logged yet. Please add some food items first, then I can give you recommendations!', 'bot');
            return;
        }

        const loadingMsg = this.addMessage('📊 Analyzing your nutrition and generating personalized recommendations...', 'bot', true);

        try {
            const response = await fetch('/api/calculate-recommendation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.userRecordId,
                    user_info: this.userInfo,
                    daily_nutrition: this.dailyNutrition
                })
            });

            let result;
            if (response.headers.get('content-type')?.includes('application/json')) {
                result = await response.json();
            } else {
                const text = await response.text();
                if (!response.ok) {
                    this.updateMessage(loadingMsg, `❌ Error (${response.status}): ${text?.slice(0, 300) || 'Unable to generate recommendation'}`);
                    return;
                }
                // Fallback wrapper if server returned plain text success
                result = { recommendation: null, raw: text };
            }

            if (!response.ok) {
                this.updateMessage(loadingMsg, `❌ Error: ${result?.error || 'Unable to generate recommendation'}`);
                return;
            }

            const rec = result.recommendation;
            if (result.user_id) {
                this.userRecordId = result.user_id;
                this.saveData();
            }
            this._lastRecommendation = rec;
            const responseMsg = this.generateRecommendationResponse(rec);
            this.updateMessage(loadingMsg, responseMsg);

            // Store recommendation for further use
            sessionStorage.setItem('lastRecommendation', JSON.stringify(rec));
            this._updateProgressBars();

        } catch (error) {
            console.error('Error:', error);
            this.updateMessage(loadingMsg, '❌ Error generating recommendation. Please try again.');
        }
    }

    updateAnalyzeButtonStatus() {
        // Enable/disable button based on current state
        const hasUserInfo = this.userInfo.gender && this.userInfo.age && this.userInfo.height && this.userInfo.weight;
        const hasFoodLogged = this.dailyNutrition.carbs > 0 || this.dailyNutrition.protein > 0 || this.dailyNutrition.fat > 0 || (this.dailyNutrition.calories || 0) > 0;
        
        if (hasUserInfo && hasFoodLogged) {
            this.analyzeBtn.style.opacity = '1';
            this.analyzeBtn.style.cursor = 'pointer';
            this.analyzeBtn.title = 'Ready! Click to get recommendations';
        } else {
            this.analyzeBtn.style.opacity = '0.7';
            this.analyzeBtn.style.cursor = 'not-allowed';
            
            if (!hasUserInfo) {
                this.analyzeBtn.title = 'Please fill in your personal info first';
            } else {
                this.analyzeBtn.title = 'Please add at least one food item first';
            }
        }
    }

    generateRecommendationResponse(rec) {
        // Generate multiple recommendation solutions
        let allSolutionsHTML = '';
        const plannerId = `recipe-planner-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
        this._recipePlanners[plannerId] = { createdAt: Date.now() };
        const plannerHTML = `
<div style="margin: 18px 0 0 0; padding-top: 16px; border-top: 2px solid var(--border);">
    <div style="font-weight: 600; color: var(--text); font-size: 14px; margin-bottom: 10px;">🎯 Build Recipes From Foods You Want</div>
    <div style="font-size: 12px; color: var(--muted); margin-bottom: 10px;">Enter foods like <strong>chicken breast, broccoli, noodles</strong>. Recipe 1 will use all of them. Recipe 2-4 can use subsets to better meet your supplement needs.</div>
    <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
        <input id="${plannerId}-input" type="text" placeholder="e.g., chicken breast, broccoli, noodles" style="flex:1; min-width:240px; padding:10px 12px; border:1px solid var(--border); border-radius:8px; font-size:13px;" onkeydown="if(event.key==='Enter'){event.preventDefault(); window._chatbot.generateCustomRecipes('${plannerId}');}">
        <button type="button" class="btn-primary" style="text-decoration:none;" onclick="window._chatbot.generateCustomRecipes('${plannerId}')">Calculate Recipes</button>
    </div>
    <div id="${plannerId}-results" style="margin-top: 12px;"></div>
</div>`;
        
        if (rec.results.length > 0) {
            rec.results.forEach((result, index) => {
                const [foods, carbSup, proteinSup, fatSup, folderName] = result;
                
                let foodList = '';
                if (foods.length > 0) {
                    foodList = foods.map(f => {
                        const dimensions = f.x && f.y && f.z 
                            ? `<span style="font-size: 11px; color: var(--muted); margin-left: 8px;">(${f.x}mm × ${f.y}mm × ${f.z}mm)</span>`
                            : '';
                        return `<li>${f.name}: ${f.gram}g ${dimensions}</li>`;
                    }).join('');
                } else {
                    foodList = '<li>No specific recommendations at this time</li>';
                }

                const supplementInfo = `
<div style="margin-top: 12px; padding: 10px; background: #f0f0f0; border-left: 3px solid var(--accent); border-radius: 4px; font-size: 12px;">
    <strong>Supplement Totals:</strong><br>
    Carbs: <strong>${carbSup}g</strong> | Protein: <strong>${proteinSup}g</strong> | Fat: <strong>${fatSup}g</strong>
</div>`;

                allSolutionsHTML += `
<div style="margin: 16px 0; padding: 14px; background: #ffffff; border: 2px solid var(--accent); border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
    <div style="font-weight: 700; margin-bottom: 12px; color: var(--accent); font-size: 15px;">💡 Option ${index + 1}</div>
    <div style="margin-bottom: 12px;">
        <div style="font-weight: 600; font-size: 13px; margin-bottom: 8px; color: var(--text);">Recommended Foods:</div>
        <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: var(--text-secondary);">
            ${foodList}
        </ul>
    </div>
    ${supplementInfo}
    ${foods && foods.some(f => f.mesh) ? `<div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top: 12px;"><a href="#" class="btn-primary" style="display:inline-block; text-decoration: none;" onclick="window._chatbot.downloadStlFolder(${index}); return false;">📁 Download STL Folder</a><a href="#" class="btn-primary" style="display:inline-block; text-decoration: none;" onclick="window._chatbot.downloadStlZip(${index}); return false;">📦 Download ZIP</a></div>${folderName ? `<div style="font-size:11px; color: var(--muted); margin-top:6px;">Folder: ${folderName}</div>` : ''}` : ''}
</div>`;
            });
        } else {
            allSolutionsHTML = '<div style="padding: 14px; background: #ffffff; border: 1px solid var(--border); border-radius: 8px; text-align: center; color: var(--muted);">No specific food recommendations at this time</div>';
        }

        return `
📊 <strong style="font-size: 16px;">Your Nutrition Recommendation</strong>

<div style="margin: 14px 0; padding: 14px; background: #ffffff; border: 1px solid var(--border); border-radius: 10px;">
    <div style="margin-bottom: 14px;">
        <div style="font-weight: 600; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Daily Energy Goal</div>
        <div style="font-size: 20px; font-weight: 700; color: var(--accent);">${rec.calories} kcal</div>
    </div>
    
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="background: #f9fafb; border-bottom: 2px solid var(--border);">
                    <th style="padding: 10px 8px; text-align: left; font-weight: 600; color: var(--text);">Nutrient</th>
                    <th style="padding: 10px 8px; text-align: right; font-weight: 600; color: var(--text);">Target</th>
                    <th style="padding: 10px 8px; text-align: right; font-weight: 600; color: var(--text);">Current</th>
                    <th style="padding: 10px 8px; text-align: right; font-weight: 600; color: var(--text);">Need</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 10px 8px;">🥔 Carbs</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600;">${rec.carbohydrate_intake}g</td>
                    <td style="padding: 10px 8px; text-align: right;">${this.dailyNutrition.carbs}g</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600; color: ${rec.carbohydrate_needed <= 0 ? '#22c55e' : 'var(--accent)'};">${rec.carbohydrate_needed}g</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 10px 8px;">🍗 Protein</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600;">${rec.protein_intake}g</td>
                    <td style="padding: 10px 8px; text-align: right;">${this.dailyNutrition.protein}g</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600; color: ${rec.protein_needed <= 0 ? '#22c55e' : 'var(--accent)'};">${rec.protein_needed}g</td>
                </tr>
                <tr>
                    <td style="padding: 10px 8px;">🥑 Fat</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600;">${rec.fat_intake}g</td>
                    <td style="padding: 10px 8px; text-align: right;">${this.dailyNutrition.fat}g</td>
                    <td style="padding: 10px 8px; text-align: right; font-weight: 600; color: ${rec.fat_needed <= 0 ? '#22c55e' : 'var(--accent)'};">${rec.fat_needed}g</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>

${plannerHTML}

<div style="margin: 18px 0 0 0; padding-top: 16px; border-top: 2px solid var(--border);">
    <div style="font-weight: 600; color: var(--text); font-size: 14px; margin-bottom: 12px;">🍽️ Suggested Food Combinations</div>
    ${allSolutionsHTML}
</div>

<div style="padding: 12px; background: rgba(34, 197, 94, 0.08); border-left: 4px solid #22c55e; border-radius: 6px; font-size: 13px; color: var(--text-secondary);">
    <strong style="color: #22c55e;">✓ Keep tracking</strong> your meals to reach your daily targets! 🎯
</div>
        `;
    }

    _escapeHtml(text) {
        return String(text ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    renderCustomRecipeResults(data) {
        const recipes = Array.isArray(data.recipes) ? data.recipes : [];
        const unresolved = Array.isArray(data.unresolved_foods) ? data.unresolved_foods : [];
        const advice = data.advice || {};

        const unresolvedHtml = unresolved.length
            ? `<div style="margin-bottom: 10px; padding: 10px; background: #fff7ed; border-left: 3px solid var(--accent); border-radius: 6px; font-size: 12px; color: var(--text-secondary);"><strong>Could not resolve:</strong> ${unresolved.map(item => this._escapeHtml(item)).join(', ')}</div>`
            : '';

        const adviceFoods = Array.isArray(advice.suggested_foods) ? advice.suggested_foods : [];
        const adviceHtml = (advice.message || adviceFoods.length)
            ? `<div style="margin-bottom: 10px; padding: 10px; background: #eefbf3; border-left: 3px solid #22c55e; border-radius: 6px; font-size: 12px; color: var(--text-secondary);"><strong>Advice:</strong> ${this._escapeHtml(advice.message || 'Consider adding these foods.')}${adviceFoods.length ? ` Suggested foods: <strong>${adviceFoods.map(item => this._escapeHtml(item)).join(', ')}</strong>.` : ''}</div>`
            : '';

        const recipesHtml = recipes.length ? recipes.map((recipe, idx) => {
            const foods = (recipe.foods || []).map(food => `<li>${this._escapeHtml(food.name)}: ${food.gram}g</li>`).join('');
            const supplied = recipe.supplied || {};
            const usingAll = recipe.uses_all_requested ? '<div style="font-size:12px; color: var(--accent); margin-bottom: 8px;"><strong>Uses all requested foods</strong></div>' : '';
            return `
<div style="margin: 12px 0; padding: 12px; background: #ffffff; border: 1px solid var(--border); border-radius: 10px;">
    <div style="font-weight: 700; margin-bottom: 8px; color: var(--text); font-size: 14px;">${this._escapeHtml(recipe.title || `Recipe ${idx + 1}`)}</div>
    ${usingAll}
    <ul style="margin: 0 0 8px 0; padding-left: 20px; font-size: 13px; color: var(--text-secondary);">${foods}</ul>
    <div style="font-size: 12px; color: var(--muted); line-height: 1.5;">
        Supplies: Carbs <strong>${supplied.carbs ?? 0}g</strong>, Protein <strong>${supplied.protein ?? 0}g</strong>, Fat <strong>${supplied.fat ?? 0}g</strong><br>
        Exceed: <strong>${recipe.exceed_total ?? 0}g</strong> | Remaining gap: <strong>${recipe.shortfall_total ?? 0}g</strong>
    </div>
</div>`;
        }).join('') : '<div style="padding: 12px; background: #ffffff; border: 1px solid var(--border); border-radius: 8px; color: var(--muted); font-size: 13px;">No recipes could be calculated from those foods.</div>';

        return `${unresolvedHtml}${adviceHtml}${recipesHtml}`;
    }

    async generateCustomRecipes(plannerId) {
        const inputEl = document.getElementById(`${plannerId}-input`);
        const resultsEl = document.getElementById(`${plannerId}-results`);
        if (!inputEl || !resultsEl) return;

        const foodText = (inputEl.value || '').trim();
        if (!foodText) {
            resultsEl.innerHTML = '<div style="padding: 10px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; font-size: 12px; color: var(--text-secondary);">Please enter foods like chicken breast, broccoli, noodles.</div>';
            return;
        }

        resultsEl.innerHTML = '<div style="padding: 10px; font-size: 12px; color: var(--muted);">Calculating recipe amounts...</div>';

        try {
            const response = await fetch('/api/calculate-custom-recipes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_info: this.userInfo,
                    daily_nutrition: this.dailyNutrition,
                    food_text: foodText
                })
            });

            const data = await response.json();
            if (!response.ok) {
                resultsEl.innerHTML = `<div style="padding: 10px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; font-size: 12px; color: var(--text-secondary);">${this._escapeHtml(data.error || 'Could not calculate recipes.')}</div>`;
                return;
            }

            resultsEl.innerHTML = this.renderCustomRecipeResults(data);
        } catch (error) {
            console.error('Custom recipe generation failed:', error);
            resultsEl.innerHTML = '<div style="padding: 10px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; font-size: 12px; color: var(--text-secondary);">Could not calculate recipes right now.</div>';
        }
    }

    async downloadStlFolder(optionIndex) {
        try {
            if (!window.showDirectoryPicker) {
                await this.downloadStlZip(optionIndex);
                return;
            }

            const rec = this._lastRecommendation || JSON.parse(sessionStorage.getItem('lastRecommendation') || 'null');
            const result = rec?.results?.[optionIndex];
            if (!result) {
                this.addMessage('⚠️ No recommendation data found for this option.', 'bot');
                return;
            }

            const [foods, , , , folderHint] = result;
            const meshFiles = (foods || []).map(f => f.mesh).filter(Boolean);
            if (meshFiles.length === 0) {
                this.addMessage('⚠️ No STL files available for this option.', 'bot');
                return;
            }

            const now = new Date();
            const y = now.getFullYear();
            const m = String(now.getMonth() + 1).padStart(2, '0');
            const d = String(now.getDate()).padStart(2, '0');
            const folderName = folderHint || `${y}${m}${d}_option${optionIndex + 1}`;

            const parentHandle = await window.showDirectoryPicker();
            const folderHandle = await parentHandle.getDirectoryHandle(folderName, { create: true });

            for (const meshName of meshFiles) {
                const url = `/download-stl/${encodeURIComponent(meshName)}`;
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`Failed to download ${meshName}: ${response.status}`);
                }
                const blob = await response.blob();
                const fileHandle = await folderHandle.getFileHandle(meshName, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(blob);
                await writable.close();
            }

            this.addMessage(`✅ Saved ${meshFiles.length} STL files to folder <strong>${folderName}</strong>.`, 'bot');
        } catch (error) {
            const isAbort = String(error?.name || '').toLowerCase() === 'aborterror';
            if (!isAbort) {
                console.error('Folder save failed:', error);
                this.addMessage(`❌ Failed to save STL folder: ${error?.message || 'Unknown error'}`, 'bot');
            }
        }
    }

    async downloadStlZip(optionIndex) {
        const rec = this._lastRecommendation || JSON.parse(sessionStorage.getItem('lastRecommendation') || 'null');
        const result = rec?.results?.[optionIndex];
        if (!result) {
            this.addMessage('⚠️ No recommendation data found for this option.', 'bot');
            return;
        }

        const [foods, , , , folderHint] = result;
        const meshFiles = (foods || []).map(f => f.mesh).filter(Boolean);
        if (meshFiles.length === 0) {
            this.addMessage('⚠️ No STL files available for this option.', 'bot');
            return;
        }

        const now = new Date();
        const y = now.getFullYear();
        const m = String(now.getMonth() + 1).padStart(2, '0');
        const d = String(now.getDate()).padStart(2, '0');
        const folderName = folderHint || `${y}${m}${d}_option${optionIndex + 1}`;

        this.addMessage('📦 Preparing ZIP download for your device...', 'bot');

        try {
            const response = await fetch('/download-stl-zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    files: meshFiles,
                    folder_name: folderName
                })
            });

            if (!response.ok) {
                let errorText = 'ZIP download failed.';
                try {
                    const errJson = await response.json();
                    errorText = errJson.error || errorText;
                } catch (_) {}
                throw new Error(errorText);
            }

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = `${folderName}.zip`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(downloadUrl);

            this.addMessage(`✅ Download started: <strong>${folderName}.zip</strong>`, 'bot');
        } catch (error) {
            console.error('ZIP download failed:', error);
            this.addMessage(`❌ Could not download ZIP: ${error?.message || 'Unknown error'}`, 'bot');
        }
    }

    undoLast() {
        if (this.conversationHistory.length === 0) {
            this.addMessage('⚠️ Nothing to undo. No food entries found.', 'bot');
            return;
        }

        // Get the last entry
        const lastEntry = this.conversationHistory.pop();
        const lastNutrition = lastEntry.nutrition;

        // Subtract from daily totals
        this.dailyNutrition.carbs -= lastNutrition.carbs || 0;
        this.dailyNutrition.protein -= lastNutrition.protein || 0;
        this.dailyNutrition.fat -= lastNutrition.fat || 0;
        this.dailyNutrition.calories = (this.dailyNutrition.calories || 0) - (lastNutrition.calories || 0);

        // Ensure no negative values
        this.dailyNutrition.carbs = Math.max(0, this.dailyNutrition.carbs);
        this.dailyNutrition.protein = Math.max(0, this.dailyNutrition.protein);
        this.dailyNutrition.fat = Math.max(0, this.dailyNutrition.fat);
        this.dailyNutrition.calories = Math.max(0, this.dailyNutrition.calories);

        // Round to 2 decimal places
        this.dailyNutrition.carbs = Math.round(this.dailyNutrition.carbs * 100) / 100;
        this.dailyNutrition.protein = Math.round(this.dailyNutrition.protein * 100) / 100;
        this.dailyNutrition.fat = Math.round(this.dailyNutrition.fat * 100) / 100;
        this.dailyNutrition.calories = Math.round(this.dailyNutrition.calories * 10) / 10;

        this.updateDisplay();
        this.saveData();
        this.updateAnalyzeButtonStatus();

        this.addMessage(`↶ Undone: <strong>${lastNutrition.food_name}</strong> (${lastNutrition.quantity}${lastNutrition.unit}) has been removed from your daily totals.`, 'bot');
    }

    clearAll() {
        if (this.dailyNutrition.carbs === 0 && this.dailyNutrition.protein === 0 && this.dailyNutrition.fat === 0 && (this.dailyNutrition.calories || 0) === 0) {
            this.addMessage('⚠️ Nothing to clear. No food entries found.', 'bot');
            return;
        }
        // Show custom confirm modal
        this.clearConfirmModal.style.display = 'flex';
    }

    _closeClearModal() {
        this.clearConfirmModal.style.display = 'none';
    }

    _confirmClear() {
        this._closeClearModal();
        // Snapshot for undo
        this._clearSnapshot = {
            dailyNutrition: { ...this.dailyNutrition },
            conversationHistory: [...this.conversationHistory]
        };
        this.dailyNutrition = { carbs: 0, protein: 0, fat: 0, calories: 0 };
        this.conversationHistory = [];
        this.updateDisplay();
        this.saveData();
        this.updateAnalyzeButtonStatus();

        // Add message with inline undo button
        const msgId = 'undo-clear-msg-' + Date.now();
        this.addMessage(
            `✨ All food entries cleared. <button id="${msgId}" ` +
            `style="margin-left:8px;padding:3px 10px;font-size:12px;cursor:pointer;` +
            `background:#ff7a3d;color:#fff;border:none;border-radius:6px;" ` +
            `onclick="window._chatbot.undoClear('${msgId}')">↩ Undo</button>`,
            'bot'
        );

        // Auto-expire undo after 10 seconds
        this._clearUndoTimer = setTimeout(() => {
            this._clearSnapshot = null;
            const btn = document.getElementById(msgId);
            if (btn) btn.remove();
        }, 10000);
    }

    undoClear(msgId) {
        if (!this._clearSnapshot) return;
        clearTimeout(this._clearUndoTimer);
        this.dailyNutrition = this._clearSnapshot.dailyNutrition;
        this.conversationHistory = this._clearSnapshot.conversationHistory;
        this._clearSnapshot = null;
        this.updateDisplay();
        this.saveData();
        this.updateAnalyzeButtonStatus();
        // Remove the undo message entirely
        const btn = document.getElementById(msgId);
        if (btn) btn.closest('.message') && btn.closest('.message').remove();
        this.addMessage('↩ Cleared entries have been restored!', 'bot');
    }

    resetDaily() {
        this.dailyNutrition = { carbs: 0, protein: 0, fat: 0, calories: 0 };
        this.conversationHistory = [];
        this.updateDisplay();
        this.saveData();
        this.addMessage('✨ Daily tracker has been reset. Start tracking your meals!', 'bot');
    }

    // ─── Calendar ─────────────────────────────────────────────────────────────

    async openCalendar() {
        if (!this.currentUserId) {
            alert('Please select a user account first.');
            return;
        }
        // Persist today's intake before showing history
        await this.saveDailyIntakeToBackend();

        try {
            const res = await fetch(`/api/daily-intake/${this.currentUserId}`);
            if (!res.ok) throw new Error('Failed to load history');
            const data = await res.json();
            this.calendarHistory = data.history || [];
        } catch (err) {
            console.error('[Calendar] Failed to load history:', err);
            this.calendarHistory = [];
        }

        const now = new Date();
        this.calendarYear = now.getFullYear();
        this.calendarMonth = now.getMonth(); // 0-indexed

        this.calendarModal.style.display = 'flex';
        this.renderCalendar();
    }

    closeCalendar() {
        this.calendarModal.style.display = 'none';
        const detail = document.getElementById('cal-day-detail');
        if (detail) detail.style.display = 'none';
    }

    renderCalendar() {
        const label = document.getElementById('cal-month-label');
        const grid = document.getElementById('cal-grid');
        const calView = document.getElementById('cal-calendar-view');
        const detail = document.getElementById('cal-day-detail');
        if (!grid) return;

        // Show calendar grid, hide detail
        if (calView) calView.style.display = '';
        if (detail) detail.style.display = 'none';
        detail.innerHTML = '';

        const monthNames = ['January','February','March','April','May','June',
                            'July','August','September','October','November','December'];
        label.textContent = `${monthNames[this.calendarMonth]} ${this.calendarYear}`;

        // Build lookup map: date-string → history entry
        const histMap = {};
        (this.calendarHistory || []).forEach(e => { histMap[e.date] = e; });

        // First weekday of month (0=Sun)
        const firstDay = new Date(this.calendarYear, this.calendarMonth, 1).getDay();
        const daysInMonth = new Date(this.calendarYear, this.calendarMonth + 1, 0).getDate();
        const todayStr = new Date().toISOString().slice(0, 10);

        grid.innerHTML = '';

        // Leading empty cells
        for (let i = 0; i < firstDay; i++) {
            const blank = document.createElement('div');
            blank.className = 'cal-cell cal-cell-empty';
            grid.appendChild(blank);
        }

        for (let d = 1; d <= daysInMonth; d++) {
            const dateStr = `${this.calendarYear}-${String(this.calendarMonth + 1).padStart(2,'0')}-${String(d).padStart(2,'0')}`;
            const entry = histMap[dateStr];
            const cell = document.createElement('div');
            cell.className = 'cal-cell';

            if (dateStr === todayStr) cell.classList.add('cal-cell-today');

            const dayNum = document.createElement('div');
            dayNum.className = 'cal-cell-day';
            dayNum.textContent = d;
            cell.appendChild(dayNum);

            if (entry) {
                const n = entry.nutrition || {};
                const kcal = Math.round((n.calories || 0) || ((n.carbs || 0) * 4 + (n.protein || 0) * 4 + (n.fat || 0) * 9));
                const kcalDiv = document.createElement('div');
                kcalDiv.className = 'cal-cell-kcal';
                kcalDiv.textContent = `${kcal} kcal`;
                cell.appendChild(kcalDiv);

                // Colour cell based on fill vs recommended
                let pct = null;
                if (entry.recommended && entry.recommended.calories) {
                    pct = kcal / entry.recommended.calories;
                }
                if (pct === null) {
                    cell.classList.add('cal-cell-has-data');
                    cell.title = 'No target saved for this day';
                } else if (pct >= 0.9 && pct <= 1.1) {
                    cell.classList.add('cal-cell-good');
                    cell.title = 'Healthy range';
                } else if (pct >= 0.75 && pct < 0.9) {
                    cell.classList.add('cal-cell-fair');
                    cell.title = 'Slightly below target';
                } else if (pct < 0.75) {
                    cell.classList.add('cal-cell-low');
                    cell.title = 'Too low';
                } else if (pct > 1.1 && pct <= 1.25) {
                    cell.classList.add('cal-cell-high');
                    cell.title = 'Slightly above target';
                } else {
                    cell.classList.add('cal-cell-excess');
                    cell.title = 'Too high';
                }

                cell.style.cursor = 'pointer';
                cell.addEventListener('click', () => this.showDayDetail(dateStr, entry));
            }

            grid.appendChild(cell);
        }

        // Wire prev/next buttons (re-attach cleanly)
        const prevBtn = document.getElementById('cal-prev');
        const nextBtn = document.getElementById('cal-next');
        const todayBtn = document.getElementById('cal-today');
        prevBtn.onclick = () => {
            this.calendarMonth--;
            if (this.calendarMonth < 0) { this.calendarMonth = 11; this.calendarYear--; }
            this.renderCalendar();
        };
        nextBtn.onclick = () => {
            this.calendarMonth++;
            if (this.calendarMonth > 11) { this.calendarMonth = 0; this.calendarYear++; }
            this.renderCalendar();
        };
        if (todayBtn) {
            todayBtn.onclick = () => this.goToTodayInCalendar();
        }
    }

    goToTodayInCalendar() {
        const now = new Date();
        this.calendarYear = now.getFullYear();
        this.calendarMonth = now.getMonth();
        this.renderCalendar();
    }

    showDayDetail(dateStr, entry) {
        const detail = document.getElementById('cal-day-detail');
        const calView = document.getElementById('cal-calendar-view');
        if (!detail) return;

        const n = entry.nutrition || {};
        const r = entry.recommended || {};

        const actualKcal = Math.round((n.carbs || 0) * 4 + (n.protein || 0) * 4 + (n.fat || 0) * 9);
        const recKcal = r.calories || null;

        const fmt = (v) => Number(v || 0).toFixed(1);
        const pctBar = (actual, target) => {
            if (!target) return '';
            const raw = Math.round((actual / target) * 100);
            if (raw <= 100) {
                const cls = raw <= 80 ? 'bar-low' : 'bar-mid';
                return `<div class="cal-bar-wrap"><div class="cal-bar-fill ${cls}" style="width:${raw}%;"></div></div>`;
            }
            // Split: base + excess
            const baseW   = (100 / raw * 100).toFixed(2);
            const excessW = ((raw - 100) / raw * 100).toFixed(2);
            const baseCls = raw <= 120 ? 'bar-over' : 'bar-danger';
            return `<div class="cal-bar-wrap">` +
                `<div class="cal-bar-fill ${baseCls} has-excess" style="width:${baseW}%;"></div>` +
                `<div class="cal-bar-excess" style="width:${excessW}%;"></div>` +
                `</div>`;
        };

        const row = (icon, lbl, actual, target, unit) => `
            <div class="cal-detail-row">
                <span class="cal-detail-label">${icon} ${lbl}</span>
                <span class="cal-detail-values">${fmt(actual)}${unit}${target ? ` / ${fmt(target)}${unit}` : ''}</span>
            </div>
            ${pctBar(actual, target)}`;

        const [y, m, d] = dateStr.split('-');
        const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        const displayDate = `${monthNames[parseInt(m)-1]} ${parseInt(d)}, ${y}`;

        detail.innerHTML = `
            <div class="cal-detail-header">
                <button type="button" class="cal-back-btn" onclick="window.chatbot.backToCalendar()">&#8592; Back</button>
                <strong>${displayDate}</strong>
            </div>
            <div class="cal-detail-body">
                <div class="cal-detail-row" style="margin-bottom:6px;">
                    <span class="cal-detail-label">&#128293; Calories</span>
                    <span class="cal-detail-values" style="font-weight:700;font-size:15px;">${actualKcal} kcal${recKcal ? ` / ${Math.round(recKcal)} kcal` : ''}</span>
                </div>
                ${recKcal ? pctBar(actualKcal, recKcal) : ''}
                <div class="cal-detail-divider"></div>
                ${row('&#127828;', 'Carbs',   n.carbs   || 0, r.carbs   || null, 'g')}
                ${row('&#127831;', 'Protein', n.protein || 0, r.protein || null, 'g')}
                ${row('&#129361;', 'Fat',     n.fat     || 0, r.fat     || null, 'g')}
            </div>`;

        // Swap views
        if (calView) calView.style.display = 'none';
        detail.style.display = 'block';
    }

    backToCalendar() {
        const detail = document.getElementById('cal-day-detail');
        const calView = document.getElementById('cal-calendar-view');
        if (detail) detail.style.display = 'none';
        if (calView) calView.style.display = '';
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new NutritionChatbot();
    window._chatbot = window.chatbot;
});
