package com.spring.biz.user.impl;

import java.util.List;


import com.spring.biz.user.UserVO;


public class UserServiceImpl implements UserService{

	private UserDAO userDAO;
	
	//set메서드로 구현
	public void setUserDAO(UserDAO userDAO) {
		this.userDAO = userDAO;
	}
	
	
	@Override
	public void insertUser(UserVO vo) {
		userDAO.insertUser(vo);
		
	}

	@Override
	public void updateUser(UserVO vo) {
		userDAO.updateUser(vo);
		
	}

	@Override
	public void deleteUser(UserVO vo) {
		userDAO.deleteUser(vo);
		
	}

	@Override
	public UserVO getUser(UserVO vo) {
		return userDAO.getUser(vo);
	}

	@Override
	public List<UserVO> getUserList() {
		return userDAO.getUserList();
	}

	@Override
	public UserVO checkuserLogin(UserVO vo) {
		return userDAO.checkUserLogin(vo);
	}

}
