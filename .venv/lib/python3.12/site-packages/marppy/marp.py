# marp.py

import numpy as np
from apexpy import Apex

RE = 6371.
hR = 0.

class Marp(Apex):
    """
    Performs coordinate conversions and base vector calculations.  Inherets
    :class:`apexpy.Apex` class.

    Parameters
    ----------
    date : float, :class:`dt.date`, or :class:`dt.datetime`, optional
        Determines which IGRF coefficients are used in conversions. Uses
        current date as default.  If float, use decimal year.
    refh : float, optional
        Reference height in km for apex coordinates (the field lines are mapped
        to this height)
    datafile : str, optional
        Path to custom coefficient file
    pole : list, optional
        Location of the new northern pole in Apex or geodetic coordinates plus
        the rotation angle around that pole (all in degrees)
    null : list, optional
        Location of the new null island ((0,0) point) in Apex or geodetic
        coordinates plus the bearing angle of new north (all in degrees)
    alt : float, optional
        Altitude of MARP origin location if it is specified in geodetic
        coordinates (defaults to the refence height)
    coords : str, optional
        Coordinate system MARP origin is specified in

    Attributes
    ----------
    year : float
        Decimal year used for the IGRF model
    refh : float
        Reference height in km for apex coordinates
    datafile : str
        Path to coefficient file
    lam0 : float
        Apex latitude of the MARP north pole
    phi0 : float
        Apex longitude of the MARP north pole
    tau0 : float
        Final rotation about MARP north pole
    R : (3,3) ndarray
        Rotation matrix to transform from Apex to MARP

    Notes
    -----
    The calculations use IGRF-13 with coefficients from 1900 to 2025 [1]_.
    The geodetic reference ellipsoid is WGS84.

    References
    ----------
    .. [1] Thebault, E. et al. (2015), International Geomagnetic Reference
           Field: the 12th generation, Earth, Planets and Space, 67(1), 79,
           :doi:`10.1186/s40623-015-0228-9`.
    """
    def __init__(self, date=None, refh=0., datafile=None, pole=None, null=None, alt=None, coords='apex'):

        super(Marp, self).__init__(date=date, refh=refh, datafile=datafile)

        if not alt:
            alt = refh

        if pole:
            if coords == 'geo':
                pole[0], pole[1] = self.geo2apex(pole[0], pole[1], alt)
            lam0, phi0, tau0 = pole
        elif null:
            if coords == 'geo':
                f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.basevectors_apex(null[0], null[1], alt)
                null[0], null[1] = self.geo2apex(null[0], null[1], alt)
                brg = np.array([np.sin(null[2]*np.pi/180.), np.cos(null[2]*np.pi/180.)])
                null[2] = np.sign(np.cross(brg, -e2[:2]))*np.arccos(np.dot(brg, -e2[:2])/np.linalg.norm(e2[:2]))*180./np.pi
            lam0, phi0, tau0 = self.null2pole(null)

        self.lam0 = lam0
        self.phi0 = phi0
        self.tau0 = tau0  # tau = angle between original pole and new null

        self.R = self.rotation_matrix(lam0, phi0, tau0)


    def null2pole(self, null):
        """
        Calculate pole location from null location.
        This makes use of the Haversine formulation and Law of Spherical
        Cosines.

        Parameters
        ----------
        null : list
            Location of the null island ((0,0) point) in Apex coordinates plus
            the bearing angle of north (all in degrees)

        Returns
        -------
        pole : list
            Location of the northern pole in Apex coordinates plus the rotation
            angle around that pole (all in degrees)

        """

        lam1, phi1, beta = np.array(null)*np.pi/180.

        # Haversine formula
        lam2 = np.arcsin(np.cos(lam1)*np.cos(beta))
        phi2 = phi1 + np.arctan2(np.sin(beta)*np.cos(lam1), -np.sin(lam1)*np.sin(lam2))

        # Use spherical law of cosines for this
        tau = np.arcsin(np.sin(np.pi/2-lam1)/np.sin(np.pi/2-lam2)*np.sin(beta)) - np.pi

        return lam2*180./np.pi, phi2*180./np.pi, tau*180./np.pi

    def rotation_matrix(self, lam0, phi0, tau0):
        """
        Calculate the Rotation Matrix.

        Parameters
        ----------
        lam0 : float
            Apex latitude of the MARP north pole
        phi0 : float
            Apex longitude of the MARP north pole
        tau0 : float
            Final rotation about MARP north pole

        Returns
        -------
        R : (3,3) ndarray
            Rotation matrix to transform from Apex to MARP

        """

        lam0 = lam0*np.pi/180.
        phi0 = phi0*np.pi/180.
        tau0 = tau0*np.pi/180.

        Rtau = np.array([[np.cos(tau0), np.sin(tau0), 0.], [-np.sin(tau0), np.cos(tau0), 0.], [0., 0., 1.]])
        Rlam = np.array([[np.sin(lam0), 0., -np.cos(lam0)], [0., 1., 0.], [np.cos(lam0), 0., np.sin(lam0)]])
        Rphi = np.array([[np.cos(phi0), np.sin(phi0), 0.], [-np.sin(phi0), np.cos(phi0), 0.], [0., 0., 1.]])

        R = np.einsum('ij,jk,kl->il', Rtau, Rlam, Rphi)

        return R


    def apex2marp(self, alat, alon):
        """
        Converts Apex to MARP coordinates.

        Parameters
        ----------
        alat : array_like
            Apex latitude
        alon : array_like
            Apex longitude

        Returns
        -------
        mlat : ndarray or float
            MARP latitude
        mlon : ndarray or float
            MARP longitude
        """

        lamA = alat*np.pi/180.
        phiA = alon*np.pi/180.

        xA = np.cos(lamA)*np.cos(phiA)
        yA = np.cos(lamA)*np.sin(phiA)
        zA = np.sin(lamA)

        rA = np.array([xA, yA, zA]).T
        rM = np.einsum('ij,...j->...i', self.R, rA).T
        xM, yM, zM = rM

        phiM = np.arctan2(yM, xM)
        lamM = np.arcsin(zM)

        mlat = lamM*180./np.pi
        mlon = phiM*180./np.pi

        return mlat, mlon

    def marp2apex(self, mlat, mlon):
        """
        Converts MARP to Apex coordinates.

        Parameters
        ----------
        mlat : ndarray or float
            MARP latitude
        mlon : ndarray or float
            MARP longitude

        Returns
        -------
        alat : array_like
            Apex latitude
        alon : array_like
            Apex longitude
        """

        lamM = mlat*np.pi/180.
        phiM = mlon*np.pi/180.

        xM = np.cos(lamM)*np.cos(phiM)
        yM = np.cos(lamM)*np.sin(phiM)
        zM = np.sin(lamM)

        rM = np.array([xM, yM, zM]).T
        rA = np.einsum('ij,...j->...i', self.R.T, rM).T
        xA, yA, zA = rA

        phiA = np.arctan2(yA, xA)
        lamA = np.arcsin(zA)

        alat = lamA*180./np.pi
        alon = phiA*180./np.pi

        return alat, alon

    def geo2marp(self, glat, glon, height):
        """
        Converts Geodetic to MARP coordinates.

        Parameters
        ----------
        glat : ndarray or float
            Geodetic latitude
        glon : ndarray or float
            Geodetic longitude
        height : array_like
            Altitude in km

        Returns
        -------
        mlat : ndarray or float
            MARP latitude
        mlon : ndarray or float
            MARP longitude
        """

        alat, alon = self.geo2apex(glat, glon, height)
        mlat, mlon = self.apex2marp(alat, alon)

        return mlat, mlon

    def marp2geo(self, mlat, mlon, height):
        """
        Converts MARP to Geodetic coordinates.

        Parameters
        ----------
        mlat : ndarray or float
            MARP latitude
        mlon : ndarray or float
            MARP longitude
        height : array_like
            Altitude in km

        Returns
        -------
        glat : ndarray or float
            Geodetic latitude
        glon : ndarray or float
            Geodetic longitude
        err : ndarray or float
            Error returned by :class:`apexpy.Apex.apex2geo`
        """

        alat, alon = self.marp2apex(mlat, mlon)
        glat, glon, err = self.apex2geo(alat, alon, height)

        return glat, glon, err

    def basevectors_marp(self, lat, lon, height, coords='geo'):
        """
        Get MARP base vectors d1, d2, d3 and e1, e2, e3 at the specified
        coordinates.

        Parameters
        ----------
        lat : (N,) array_like or float
            Latitude
        lon : (N,) array_like or float
            Longitude
        height : (N,) array_like or float
            Altitude in km
        coords : {'geo', 'apex'}, optional
            Input coordinate system

        Returns
        -------
        d1 : (3, N) or (3,) ndarray
            MARP base vector normal to contours of constant PhiM
        d2 : (3, N) or (3,) ndarray
            MARP base vector normal to contours of constant LamM
        d3 : (3, N) or (3,) ndarray
            MARP base vector normal to contours of constant V0
        e1 : (3, N) or (3,) ndarray
            MARP base vector tangent to contours of constant LamM
        e2 : (3, N) or (3,) ndarray
            MARP base vector tangent to contours of constant PhiM
        e3 : (3, N) or (3,) ndarray
            MARP base vector tangent to magnetic field
        """

        if coords == 'geo':
            glat = lat
            glon = lon
            alat, alon = self.geo2apex(glat, glon, height)
            mlat, mlon = self.apex2marp(alat, alon)

        if coords == 'apex':
            alat = lat
            alon = lon
            glat, glon, _ = self.apex2geo(alat, alon, height)
            mlat, mlon = self.apex2marp(alat, alon)

        lM = np.asarray(mlat)*np.pi/180.
        pM = np.asarray(mlon)*np.pi/180.
        lA = np.asarray(alat)*np.pi/180.
        pA = np.asarray(alon)*np.pi/180.

        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = self.basevectors_apex(glat, glon, height)

        sinI = 2*np.sin(lA)/np.sqrt(4-3*np.cos(lA)**2)

        # Calculate contravarient base vectors
        xM = np.cos(lM)*np.cos(pM)
        yM = np.cos(lM)*np.sin(pM)
        zM = np.sin(lM)

        P1 = np.array([[-np.sin(pA)*np.cos(lA), -np.cos(pA)*np.sin(lA)], [np.cos(pA)*np.cos(lA), -np.sin(pA)*np.sin(lA)], [np.zeros(lA.shape), np.cos(lA)]])
        P2 = np.array([[-yM/(xM**2+yM**2), xM/(xM**2+yM**2), np.zeros(xM.shape)], [np.zeros(xM.shape), np.zeros(xM.shape), 1./np.sqrt(1-zM**2)]])

        dMdA = np.einsum('ij...,jk,kl...->il...', P2, self.R, P1)

        dpMdpA = dMdA[0,0]
        dpMdlA = dMdA[0,1]
        dlMdpA = dMdA[1,0]
        dlMdlA = dMdA[1,1]

        d1M = (d1/np.cos(lA)*dpMdpA - d2/sinI*dpMdlA)*np.cos(lA)/np.sqrt(dpMdpA*dlMdlA-dpMdlA*dlMdpA)
        d2M = -(d1/np.cos(lA)*dlMdpA - d2/sinI*dlMdlA)*sinI/np.sqrt(dpMdpA*dlMdlA-dpMdlA*dlMdpA)
        d3M = d3

        # Calculate covarient base vectors
        xA = np.cos(lA)*np.cos(pA)
        yA = np.cos(lA)*np.sin(pA)
        zA = np.sin(lA)

        P1 = np.array([[-np.sin(pM)*np.cos(lM), -np.cos(pM)*np.sin(lM)], [np.cos(pM)*np.cos(lM), -np.sin(pM)*np.sin(lM)], [np.zeros(lM.shape), np.cos(lM)]])
        P2 = np.array([[-yA/(xA**2+yA**2), xA/(xA**2+yA**2), np.zeros(xA.shape)], [np.zeros(xA.shape), np.zeros(xA.shape), 1./np.sqrt(1-zA**2)]])

        dAdM = np.einsum('ij...,jk,kl...->il...', P2, self.R.T, P1)

        dpAdpM = dAdM[0,0]
        dpAdlM = dAdM[0,1]
        dlAdpM = dAdM[1,0]
        dlAdlM = dAdM[1,1]

        e1M = (e1*np.cos(lA)*dpAdpM - e2*sinI*dlAdpM)/np.cos(lA)/np.sqrt(dpAdpM*dlAdlM-dlAdpM*dpAdlM)
        e2M = -(e1*np.cos(lA)*dpAdlM - e2*sinI*dlAdlM)/sinI/np.sqrt(dpAdpM*dlAdlM-dlAdpM*dpAdlM)
        e3M = e3

        return d1M, d2M, d3M, e1M, e2M, e3M
