package com.spring.biz.user.impl;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;

import com.spring.biz.common.JDBCUtil;
import com.spring.biz.user.UserVO;

public class UserDAO {
	
	private Connection conn = null;
	private PreparedStatement pstmt = null;
	private ResultSet rs = null;
	
	private final String USER_INSERT = 
			"insert into users values (?, ?, ?, ?)";
	private final String USER_UPDATE = 
			"update users set name=?, role=? where id=?";
	private final String USER_DELETE = 
			"delete from users where id=?";
	private final String USER_GET = 
			"select * from users where id=?";
	private final String USER_LIST = 
			"select * from users order by id desc";
	private final String USER_CHECK =
			"select * from users where id=? and password=?";
	
	public void insertUser(UserVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_INSERT);
			
			pstmt.setString(1, vo.getId());
			pstmt.setString(2, vo.getPassword());
			pstmt.setString(3, vo.getName());
			pstmt.setString(4, vo.getRole());
			
			pstmt.executeUpdate();
			
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public void updateUser(UserVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_UPDATE);
			//update users set name=?, role=? where id=?
			
			pstmt.setString(1, vo.getName());
			pstmt.setString(2, vo.getRole());
			pstmt.setString(3, vo.getId());
			
			pstmt.executeUpdate();
			
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public void deleteUser(UserVO vo) {
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_DELETE);
			//delete from users where id=?
			pstmt.setString(1, vo.getId());
			pstmt.executeUpdate();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(pstmt, conn);
		}
	}
	
	public UserVO getUser(UserVO vo) {
		UserVO user = null;
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_GET);
			//select * from users where id=?
			pstmt.setString(1, vo.getId());
			rs = pstmt.executeQuery();
			if(rs.next()) {
				user = new UserVO();
				user.setId(rs.getString("id"));
				user.setPassword(rs.getString("password"));
				user.setName(rs.getString("name"));
				user.setRole(rs.getString("role"));
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(rs, pstmt, conn);
		}
		return user;
	}
	
	public List<UserVO> getUserList(){
		List<UserVO> userList = new ArrayList<UserVO>();
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_LIST);
			rs = pstmt.executeQuery();
			
			while(rs.next()) {
				UserVO user = new UserVO();
				user.setId(rs.getString("id"));
				user.setPassword(rs.getString("password"));
				user.setName(rs.getString("name"));
				user.setRole(rs.getString("role"));
				
				userList.add(user);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			JDBCUtil.close(rs, pstmt, conn);
		}
		return userList;
	}
	
	public UserVO checkUserLogin(UserVO vo) {
		UserVO user = null;
		try {
			conn = JDBCUtil.getConnection();
			pstmt = conn.prepareStatement(USER_CHECK);
			//select * from users where id=? and password=?
			pstmt.setString(1, vo.getId());
			pstmt.setString(2, vo.getPassword());
			rs = pstmt.executeQuery();
			if(rs.next()) {
				user = new UserVO();
				user.setId(rs.getString("id"));
				user.setPassword(rs.getString("password"));
				user.setName(rs.getString("name"));
				user.setRole(rs.getString("role"));
			}
		} catch (Exception e) {
			// TODO: handle exception
		} finally {
			JDBCUtil.close(rs, pstmt, conn);
		}
		return user;
	}
}
